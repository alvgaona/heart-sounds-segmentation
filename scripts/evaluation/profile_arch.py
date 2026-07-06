"""Profile emitter architectures for the C8 deployability column: params, size, latency, real-time factor.

Compares the BiLSTM and xLSTM CRF emitters at matched settings. Latency is the emitter forward
(features -> emissions) on CPU for one 2 s window; the constrained decoder is architecture-independent
so it is excluded from the comparison. int8 size uses dynamic quantization of the Linear/LSTM layers.

    pixi run python scripts/evaluation/profile_arch.py
"""

import argparse
import io
import time

import torch
from torch import nn

from hss.model.segmenter_crf import HeartSoundSegmenterCRF
from hss.model.segmenter_xlstm import HeartSoundSegmenterXLSTMCRF


CONFIGS = [
    ("BiLSTM-240", lambda: HeartSoundSegmenterCRF(input_size=44, hidden_size=240)),
    ("xLSTM-240", lambda: HeartSoundSegmenterXLSTMCRF(input_size=44, hidden_size=240, num_heads=4)),
    ("xLSTM-336", lambda: HeartSoundSegmenterXLSTMCRF(input_size=44, hidden_size=336, num_heads=4)),
]


def serialized_mb(module: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def bench_latency(net: nn.Module, x: torch.Tensor, iters: int, warmup: int) -> float:
    """Mean seconds for one emitter forward (emissions), on CPU."""
    net.eval()
    with torch.no_grad():
        for _ in range(warmup):
            net(x)
        start = time.perf_counter()
        for _ in range(iters):
            net(x)
    return (time.perf_counter() - start) / iters


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-len", type=int, default=100, help="frames per window (100 = 2 s @ 50 Hz)")
    ap.add_argument("--rate", type=float, default=50.0, help="frame rate (Hz) for the real-time factor")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(0)
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"  # macOS/ARM default is often unset
    x = torch.randn(1, args.seq_len, 44)
    window_s = args.seq_len / args.rate

    print(f"CPU emitter forward | {args.seq_len} frames ({window_s:.2f}s @ {args.rate:g}Hz) | {args.iters} iters\n")
    header = f"{'config':<12} {'params':>10} {'fp32 MB':>8} {'int8 MB':>8} {'ms/window':>10} {'RTF':>8}"
    print(header)
    print("-" * len(header))
    for name, build in CONFIGS:
        net = build()
        params = sum(p.numel() for p in net.parameters())
        fp32 = serialized_mb(net)
        try:
            qnet = torch.quantization.quantize_dynamic(net, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
            int8 = f"{serialized_mb(qnet):.2f}"
        except RuntimeError:
            int8 = "n/a"  # no quantized engine on this platform
        sec = bench_latency(net, x, args.iters, args.warmup)
        rtf = window_s / sec  # >1 means faster than real time
        print(f"{name:<12} {params:>10,} {fp32:>8.2f} {int8:>8} {sec * 1e3:>10.2f} {rtf:>8.1f}x")


if __name__ == "__main__":
    main()
