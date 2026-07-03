#!/usr/bin/env python3
"""Re-score trained CV checkpoints under different decoders — no retraining.

Loads each fold's best checkpoint and evaluates the held-out fold under three decoders:
  - viterbi   : the model's native Viterbi (linear-chain for CRF/TCN, semi-Markov for semi_crf)
  - marginal  : argmax of the forward-backward posterior marginals
  - posterior : constrained-posterior (frame-level constrained Viterbi over the marginals -> valid cycle)

This standardizes the decode across models so encoder effects can be separated from decoder effects.
"""

import argparse
import glob
import re

import torch
from torchmetrics.classification import F1Score


DEFAULT_LOG_DIR = {"crf": "lightning_logs_crf", "tcn": "lightning_logs_tcn", "semi_crf": "lightning_logs_semi_crf"}
DECODERS = ["viterbi", "marginal", "posterior"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["crf", "tcn", "semi_crf"], required=True)
    parser.add_argument("--log-dir", default=None, help="fold checkpoint root (default per model)")
    parser.add_argument("--fsst-path", default="data/springer_fsst/springer_fsst.pt")
    parser.add_argument("--downsample", type=int, default=20)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=68)
    parser.add_argument("--batch-size", type=int, default=50, help="must match the trained batch size for CRF")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu", "mps"], default="cpu")
    return parser.parse_args()


def downsample_time(features: torch.Tensor, labels: torch.Tensor, factor: int) -> tuple[torch.Tensor, torch.Tensor]:
    if factor <= 1:
        return features, labels
    n, t, c = features.shape
    t2 = (t // factor) * factor
    nt = t2 // factor
    return features[:, :t2, :].reshape(n, nt, factor, c).mean(2), labels[:, :t2].reshape(n, nt, factor).mode(2).values


def kfold_indices(n: int, k: int, seed: int) -> list[tuple[list[int], list[int], list[int]]]:
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    folds = [perm[i::k] for i in range(k)]
    splits = []
    for i in range(k):
        rest = [idx for j, f in enumerate(folds) if j != i for idx in f]
        vs = int(0.15 * len(rest))
        splits.append((rest[vs:], rest[:vs], folds[i]))
    return splits


def latest_ckpt(log_dir: str, fold: int) -> str | None:
    paths = glob.glob(f"{log_dir}/fold_{fold}/lightning_logs/version_*/checkpoints/*.ckpt")
    if not paths:
        return None
    return max(paths, key=lambda p: int(re.search(r"version_(\d+)", p).group(1)))


def load_segmenter(model_type: str, ckpt: str, device: torch.device, batch_size: int) -> torch.nn.Module:
    if model_type == "crf":
        from hss.model.lit_model_crf import LitModelCRF

        # Infer input_size from the checkpoint (44 for FSST-only, 46 with envelope fusion) so both load.
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        input_size = state["model.lstm_1.weight_ih_l0"].shape[1]
        lit = LitModelCRF.load_from_checkpoint(
            ckpt, input_size=input_size, batch_size=batch_size, device=device, map_location=device
        )
    elif model_type == "tcn":
        from hss.model.lit_model_tcn import LitModelTCN

        lit = LitModelTCN.load_from_checkpoint(ckpt, map_location=device)
    else:
        from hss.model.lit_model_semi_crf import LitModelSemiCRF

        lit = LitModelSemiCRF.load_from_checkpoint(ckpt, device=device, map_location=device)
    return lit.to(device).eval().model


def decode(net: torch.nn.Module, x: torch.Tensor, method: str) -> torch.Tensor:
    if method == "viterbi":
        return net.decode(x)
    if method == "marginal":
        return net.marginals(x).argmax(-1)
    return net.decode_valid(x)  # posterior


def get_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(args: argparse.Namespace) -> None:
    log_dir = args.log_dir or DEFAULT_LOG_DIR[args.model]
    device = get_device(args.accelerator)
    print(f"Model: {args.model} | log-dir: {log_dir} | device: {device}")

    data = torch.load(args.fsst_path, weights_only=True, mmap=args.downsample > 1)
    features, labels = data["features"], data["labels"]
    if args.downsample > 1:
        features, labels = downsample_time(features, labels, args.downsample)
    splits = kfold_indices(len(features), args.folds, args.seed)

    results: dict[str, list[float]] = {d: [] for d in DECODERS}
    for i in range(args.folds):
        ckpt = latest_ckpt(log_dir, i)
        if ckpt is None:
            print(f"fold {i + 1}: no checkpoint found, skipping")
            continue
        try:
            net = load_segmenter(args.model, ckpt, device, args.batch_size)
        except Exception as e:  # e.g. a checkpoint still being written by a running job
            print(f"fold {i + 1}: load failed ({type(e).__name__}), skipping")
            continue
        _, _, test_idx = splits[i]
        fx, yx = features[test_idx], labels[test_idx]
        preds: dict[str, list[torch.Tensor]] = {d: [] for d in DECODERS}
        with torch.no_grad():
            for b in range(0, len(fx), args.batch_size):
                x = fx[b : b + args.batch_size].to(device)
                for d in DECODERS:
                    preds[d].append(decode(net, x, d).cpu())
        row = [f"fold {i + 1}"]
        for d in DECODERS:
            f1 = F1Score(task="multiclass", num_classes=4, average="macro")(
                torch.cat(preds[d]).reshape(-1), yx.reshape(-1)
            ).item()
            results[d].append(f1)
            row.append(f"{d}={f1:.4f}")
        print("  ".join(row))
        del net

    print("\n" + "=" * 60)
    print(f"{args.model} — macro F1 by decoder (mean ± std over folds)")
    print("=" * 60)
    for d in DECODERS:
        v = torch.tensor(results[d])
        if v.numel():
            std = v.std(unbiased=True).item() if v.numel() > 1 else 0.0
            print(f"{d:10s}: {v.mean().item():.4f} ± {std:.4f}")


if __name__ == "__main__":
    main(parse_args())
