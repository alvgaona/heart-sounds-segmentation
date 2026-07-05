#!/usr/bin/env python3
"""C8 quantization: measured F1 retention under post-training dynamic INT8 (and FP16 size). No retraining.

Dynamic INT8 (`quantize_dynamic` on nn.LSTM + nn.Linear) is the standard PTQ for recurrent nets: weights ->
int8, activations quantized per-batch at runtime, no calibration set. It quantizes the BiLSTM (the dominant
cost per the latency benchmark); the CRF's log-space math stays fp32 and runs on the (now int8-derived)
emissions — the risky interaction we MEASURE rather than assume.

Per patient fold: decode the held-out set with the fp32 model and its int8 copy; report macro / 2-class F1 and
valid-cycle for each, and ΔF1 = int8 - fp32. Also serialized size (fp32/fp16/int8) and CPU forward latency.
"""

import argparse
import io
import time

import torch
from reeval_decoders import (
    add_split_args,
    build_test_splits,
    decode_preds,
    downsample_time,
    latest_ckpt,
    load_segmenter,
    valid_cycle_fraction,
)
from torch import nn
from torchmetrics.classification import F1Score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        choices=["crf", "lstm_valid"],
        default="lstm_valid",
        help="crf = LSTM+CRF+decode_valid; lstm_valid = softmax+decode_valid (the deployable)",
    )
    p.add_argument("--log-dir", required=True)
    p.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt")
    p.add_argument("--downsample", type=int, default=20)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=68)
    p.add_argument("--batch-size", type=int, default=50)
    add_split_args(p)
    return p.parse_args()


def quantize_int8(net: nn.Module) -> nn.Module:
    torch.backends.quantized.engine = "qnnpack"  # Apple Silicon / ARM backend
    return torch.quantization.quantize_dynamic(net.to("cpu"), {nn.LSTM, nn.Linear}, dtype=torch.qint8)


def serialized_mb(net: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    return buf.tell() / 1e6


def main(args: argparse.Namespace) -> None:
    cpu = torch.device("cpu")
    data = torch.load(args.fsst_path, weights_only=True, mmap=args.downsample > 1)
    features, labels_ds = downsample_time(data["features"], data["labels"], args.downsample)
    splits = build_test_splits(len(features), args)
    macro = F1Score(task="multiclass", num_classes=4, average="macro")
    two = F1Score(task="multiclass", num_classes=2, average=None)
    print(f"Model {args.model} | {args.log_dir} | dynamic INT8 (LSTM+Linear) | FSST patient")

    acc: dict[str, list[float]] = {k: [] for k in ("f32_macro", "i8_macro", "f32_s1s", "i8_s1s", "f32_vc", "i8_vc")}
    size_f32 = size_i8 = None
    for i in range(args.folds):
        ckpt = latest_ckpt(args.log_dir, i)
        if ckpt is None:
            continue
        net = load_segmenter(args.model, ckpt, cpu, args.batch_size).eval()
        qnet = quantize_int8(net)
        if size_f32 is None:
            size_f32, size_i8 = serialized_mb(net), serialized_mb(qnet)
        _, _, test_idx = splits[i]
        fx, y = features[test_idx], labels_ds[test_idx]

        for tag, model in (("f32", net), ("i8", qnet)):
            preds = []
            with torch.no_grad():
                for b in range(0, len(fx), args.batch_size):
                    preds.append(decode_preds(model, fx[b : b + args.batch_size], args.model))
            p = torch.cat(preds)
            acc[f"{tag}_macro"].append(macro(p.reshape(-1), y.reshape(-1)).item())
            acc[f"{tag}_s1s"].append(two((p >= 2).long().reshape(-1), (y >= 2).long().reshape(-1))[0].item())
            acc[f"{tag}_vc"].append(valid_cycle_fraction(p))
        print(f"fold {i + 1}: done")

    def m(v: list[float]) -> float:
        return float(torch.tensor(v).mean())

    def s(v: list[float]) -> float:
        return float(torch.tensor(v).std(unbiased=True)) if len(v) > 1 else 0.0

    # latency: fp32 vs int8 forward on a full-recording-sized input
    x = features[:1, :, :].to(cpu)  # (1, T, C) at working rate; time the emission forward
    net = load_segmenter(args.model, latest_ckpt(args.log_dir, 0), cpu, args.batch_size).eval()
    qnet = quantize_int8(net)

    def timeit(mdl):
        for _ in range(3):
            (mdl._get_emissions(x) if hasattr(mdl, "_get_emissions") else mdl(x))
        ts = []
        for _ in range(15):
            t0 = time.perf_counter()
            (mdl._get_emissions(x) if hasattr(mdl, "_get_emissions") else mdl(x))
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[len(ts) // 2] * 1000

    with torch.no_grad():
        lat_f32, lat_i8 = timeit(net), timeit(qnet)

    print("\n" + "=" * 66)
    print(f"Post-training dynamic INT8 — {args.model} (mean ± std over folds)")
    print("=" * 66)
    d_macro = m(acc["i8_macro"]) - m(acc["f32_macro"])
    print(
        f"macro F1     fp32 {m(acc['f32_macro']):.4f} ± {s(acc['f32_macro']):.4f} | "
        f"int8 {m(acc['i8_macro']):.4f} ± {s(acc['i8_macro']):.4f} | ΔF1 {d_macro:+.4f}"
    )
    print(
        f"2-class S1+Sys  fp32 {m(acc['f32_s1s']):.4f} | int8 {m(acc['i8_s1s']):.4f} | "
        f"Δ {m(acc['i8_s1s']) - m(acc['f32_s1s']):+.4f}"
    )
    print(f"valid-cycle  fp32 {m(acc['f32_vc']):.4f} | int8 {m(acc['i8_vc']):.4f}")
    print(
        f"\nserialized size: fp32 {size_f32:.2f} MB | fp16 ~{size_f32 / 2:.2f} MB | int8 {size_i8:.2f} MB "
        f"({size_f32 / size_i8:.1f}x smaller)"
    )
    print(
        f"BiLSTM forward latency (CPU, T={features.shape[1]}): fp32 {lat_f32:.2f} ms | int8 {lat_i8:.2f} ms "
        f"({lat_f32 / lat_i8:.2f}x)"
    )


if __name__ == "__main__":
    main(parse_args())
