#!/usr/bin/env python3
"""C8 latency breakdown: feature extraction vs BiLSTM forward vs decode, per config, on CPU (batch=1).

Times each stage of the deployable continuous pipeline (whole recording in one pass) and reports ms, ms per
second of audio, and the real-time factor (RTF = inference time / audio duration; <1 = faster than real-time).
Timing is content-independent for FFT/CWT + the recurrent net, so a synthetic signal of the given length is
used. Fresh (random-weight) models — latency is architecture-driven, not weight-driven.

Stages:
  - feature extraction: FSST (STFT-based) vs WSST-nv8 (CWT-based) — the front-end cost, expected to dominate.
  - forward: 2-layer BiLSTM emissions.
  - decode: argmax (softmax) | softmax + decode_valid (constrained Viterbi over emissions) | CRF decode_valid
    (forward-backward + constrained Viterbi) | semi-CRF decode_valid (O(T*D) marginals + constrained Viterbi).
"""

import argparse
import statistics
import time

import scipy.signal
import torch

from hss.model.segmenter import HeartSoundSegmenter
from hss.model.segmenter_crf import HeartSoundSegmenterCRF
from hss.model.segmenter_semi_crf import HeartSoundSegmenterSemiCRF
from hss.transforms import FSST, WSST
from hss.utils.sequence_validator import validate_and_correct_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-seconds", type=float, default=8.0, help="Synthetic recording length")
    p.add_argument("--fs", type=int, default=1000)
    p.add_argument("--downsample", type=int, default=20, help="1000 Hz -> 50 Hz")
    p.add_argument("--num-voices", type=int, default=8)
    p.add_argument("--max-duration", type=int, default=100, help="semi-CRF cap (frames @ working rate)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=15)
    p.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: torch default)")
    return p.parse_args()


def timeit(fn, warmup: int, iters: int) -> float:
    """Median wall-clock over `iters` warm runs, in ms."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000.0


def pool(feat: torch.Tensor, factor: int) -> torch.Tensor:
    t2 = (feat.shape[0] // factor) * factor
    return feat[:t2].reshape(t2 // factor, factor, -1).mean(1)


def main(args: argparse.Namespace) -> None:
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    audio_ms = args.audio_seconds * 1000.0
    n = int(args.audio_seconds * args.fs)
    x = torch.randn(n)  # raw signal; FFT/CWT timing is content-independent

    fsst = FSST(
        args.fs,
        window=scipy.signal.get_window(("kaiser", 0.5), 128, fftbins=False),
        truncate_freq=(25, 200),
        stack=True,
    )
    wsst = WSST(args.fs, num_voices=args.num_voices, truncate_freq=(25, 200), stack=True)

    print(
        f"CPU threads: {torch.get_num_threads()} | audio: {args.audio_seconds:.1f}s ({n} samp @ {args.fs} Hz) "
        f"| rate {args.fs // args.downsample} Hz | warm×{args.warmup} med×{args.iters}"
    )

    # --- feature extraction ---
    t_fsst = timeit(lambda: fsst(x), args.warmup, args.iters)
    t_wsst = timeit(lambda: wsst(x), args.warmup, args.iters)
    feat_fsst = pool(fsst(x), args.downsample)  # (T, 44)
    feat_wsst = pool(wsst(x), args.downsample)  # (T, 48)
    xf, xw = feat_fsst[None], feat_wsst[None]
    tt = feat_fsst.shape[0]
    print(f"feature dims: FSST {feat_fsst.shape[1]}, WSST {feat_wsst.shape[1]} | T={tt} frames")

    # --- models (fresh weights; eval = dropout off) ---
    soft = HeartSoundSegmenter(input_size=xf.shape[-1]).eval()
    crf_f = HeartSoundSegmenterCRF(input_size=xf.shape[-1]).eval()
    crf_w = HeartSoundSegmenterCRF(input_size=xw.shape[-1]).eval()
    semi = HeartSoundSegmenterSemiCRF(input_size=xf.shape[-1], max_duration=args.max_duration).eval()

    # decode_valid / the softmax decodes each run the BiLSTM forward internally, so they already INCLUDE the
    # forward cost. Report the forward separately, and decode-ONLY = (full decode) - forward.
    with torch.no_grad():
        t_fwd = timeit(lambda: crf_f._get_emissions(xf), args.warmup, args.iters)
        t_wfwd = timeit(lambda: crf_w._get_emissions(xw), args.warmup, args.iters)
        t_argmax = timeit(lambda: soft(xf).argmax(-1), args.warmup, args.iters)  # forward + argmax
        t_soft = timeit(lambda: validate_and_correct_predictions(soft(xf)), args.warmup, args.iters)  # fwd + Viterbi
        t_crf_f = timeit(lambda: crf_f.decode_valid(xf), args.warmup, args.iters)  # fwd + fwd-bwd + Viterbi
        t_crf_w = timeit(lambda: crf_w.decode_valid(xw), args.warmup, args.iters)
        t_semi = timeit(lambda: semi.decode_valid(xf), max(1, args.warmup // 2), max(3, args.iters // 3))

    def row(name: str, ms: float) -> None:
        print(f"  {name:<38} {ms:8.2f} ms   {ms / args.audio_seconds:7.2f} ms/s-audio")

    print("\nStage latency (median):")
    row("FSST extraction", t_fsst)
    row(f"WSST-nv{args.num_voices} extraction", t_wsst)
    row("BiLSTM forward (FSST 44d)", t_fwd)
    row("BiLSTM forward (WSST 48d)", t_wfwd)
    print("  decode-only (excludes the forward above):")
    row("  argmax", t_argmax - t_fwd)
    row("  softmax + constrained Viterbi", t_soft - t_fwd)
    row("  CRF decode_valid (fwd-bwd + Viterbi)", t_crf_f - t_fwd)
    row("  semi-CRF decode_valid (O(T*D))", t_semi - t_fwd)

    # end-to-end = feature + full decode (decode already includes the forward).
    configs = {
        "FSST + softmax + decode_valid (deployable)": t_fsst + t_soft,
        "FSST + CRF   + decode_valid": t_fsst + t_crf_f,
        "WSST + CRF   + decode_valid": t_wsst + t_crf_w,
        "FSST + semi-CRF + decode_valid": t_fsst + t_semi,
    }
    print("\nEnd-to-end per recording  |  RTF (infer / audio; <1 = real-time):")
    for name, ms in configs.items():
        print(f"  {name:<44} {ms:8.2f} ms   RTF {ms / audio_ms:.4f}")


if __name__ == "__main__":
    main(parse_args())
