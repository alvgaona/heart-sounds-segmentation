"""Streaming latency for the causal xLSTM segmenter (Experiment B, phase 3).

Runs the causal emitter one frame at a time (O(1) state, no growth in T) and the fixed-lag constrained
decoder, then reports per-frame latency vs the frame period. Real-time means ms/frame < the frame period
(20 ms @ 50 Hz). This is the deployability number the streaming architecture is for.

    pixi run python scripts/evaluation/profile_streaming.py
"""

import argparse
import time

import torch

from hss.model.segmenter_xlstm import HeartSoundSegmenterXLSTMCRF
from hss.utils.streaming_decode import stream_decode, valid_transition_fraction


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=1000, help="frames to stream")
    ap.add_argument("--rate", type=float, default=50.0, help="frame rate (Hz)")
    ap.add_argument("--lag", type=int, default=10, help="decoder latency in frames")
    ap.add_argument("--hidden", type=int, default=240)
    ap.add_argument("--warmup", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(0)
    model = HeartSoundSegmenterXLSTMCRF(input_size=44, hidden_size=args.hidden, num_heads=4, bidirectional=False).eval()
    enc, head = model.encoder, model.linear
    frame_period_ms = 1000.0 / args.rate

    xs = torch.randn(args.frames, 44)
    emissions = torch.zeros(args.frames, 4)

    # Streaming emitter: one frame at a time, constant state.
    with torch.no_grad():
        state = enc.init_state(1)
        for i in range(min(args.warmup, args.frames)):  # warmup
            h, state = enc.step(xs[i : i + 1], state)
        state = enc.init_state(1)
        start = time.perf_counter()
        for i in range(args.frames):
            h, state = enc.step(xs[i : i + 1], state)
            emissions[i] = head(h)[0]
        emitter_ms = (time.perf_counter() - start) / args.frames * 1e3

    # Fixed-lag decode over the streamed emissions.
    start = time.perf_counter()
    path = stream_decode(emissions, lag=args.lag)
    decode_ms = (time.perf_counter() - start) / args.frames * 1e3

    per_frame = emitter_ms + decode_ms
    params = sum(p.numel() for p in model.parameters())
    print(
        f"Causal xLSTM-{args.hidden} | {params:,} params | {args.frames} frames @ {args.rate:g} Hz | lag {args.lag}\n"
    )
    print(f"  emitter step   : {emitter_ms:6.3f} ms/frame")
    print(f"  fixed-lag decode: {decode_ms:6.3f} ms/frame")
    print(f"  total          : {per_frame:6.3f} ms/frame   (frame period {frame_period_ms:.1f} ms)")
    verdict = "REAL-TIME" if per_frame < frame_period_ms else "TOO SLOW"
    print(f"  real-time factor: {frame_period_ms / per_frame:6.1f}x   {verdict}")
    print(f"  valid-cycle     : {valid_transition_fraction(path):.4f} (streamed, lag {args.lag})")


if __name__ == "__main__":
    main()
