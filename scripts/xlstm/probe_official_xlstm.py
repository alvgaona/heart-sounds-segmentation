"""Probe the official ``xlstm`` package to build an exact parity fixture (run on Lightning Studio).

We do NOT hard-code the official API — this script *discovers* it, so the parity test is written
against the exact installed version instead of a guess. It:

  1. prints the installed ``xlstm`` version,
  2. finds the pure-PyTorch mLSTM backend (``parallel_stabilized_simple`` or similar),
  3. runs a small **seeded** fixture through it, and
  4. writes inputs + output to ``xlstm_probe.json``.

Our vendored ``mlstm_parallel`` mirrors that backend's convention:
    q, k, v         : (B, NH, T, DH)
    igate/fgate     : (B, NH, T, 1)   pre-activations
    returns h       : (B, NH, T, DH)

Usage on Lightning Studio (CUDA box):
    pip install xlstm
    python probe_official_xlstm.py          # writes xlstm_probe.json next to the script
Then send back ``xlstm_probe.json`` — no repo checkout needed; this file is standalone.
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from typing import Callable

import torch


FIXTURE = dict(B=1, NH=2, T=8, DH=4, seed=1234)


def find_backend() -> tuple[str, Callable] | tuple[None, None]:
    """Locate the pure-torch mLSTM parallel backend by walking the xlstm package."""
    import xlstm

    candidates: list[tuple[str, Callable]] = []
    # 1) the known location in current releases
    for path in ("xlstm.blocks.mlstm.backends", "xlstm.mlstm.backends"):
        try:
            mod = importlib.import_module(path)
        except Exception:  # noqa: BLE001
            continue
        for name, obj in vars(mod).items():
            if callable(obj) and "parallel_stabilized" in name:
                candidates.append((f"{path}.{name}", obj))
    # 2) fall back to a full package walk if not found above
    if not candidates:
        for info in pkgutil.walk_packages(xlstm.__path__, prefix="xlstm."):
            try:
                mod = importlib.import_module(info.name)
            except Exception:  # noqa: BLE001
                continue
            for name, obj in vars(mod).items():
                if callable(obj) and "parallel_stabilized" in name:
                    candidates.append((f"{info.name}.{name}", obj))
    return candidates[0] if candidates else (None, None)


def build_fixture() -> dict[str, torch.Tensor]:
    torch.manual_seed(FIXTURE["seed"])
    b, nh, t, dh = FIXTURE["B"], FIXTURE["NH"], FIXTURE["T"], FIXTURE["DH"]
    return {
        "q": torch.randn(b, nh, t, dh, dtype=torch.float64),
        "k": torch.randn(b, nh, t, dh, dtype=torch.float64),
        "v": torch.randn(b, nh, t, dh, dtype=torch.float64),
        "igate_preact": torch.randn(b, nh, t, 1, dtype=torch.float64),
        "fgate_preact": torch.randn(b, nh, t, 1, dtype=torch.float64) + 2.0,
    }


def call_backend(fn: Callable, fx: dict[str, torch.Tensor]) -> torch.Tensor:
    """Call the discovered backend, adapting to its parameter names."""
    params = set(inspect.signature(fn).parameters)
    name_map = {
        "q": ["queries", "q"],
        "k": ["keys", "k"],
        "v": ["values", "v"],
        "igate_preact": ["igate_preact", "igate", "i"],
        "fgate_preact": ["fgate_preact", "fgate", "f"],
    }
    kwargs = {}
    for ours, aliases in name_map.items():
        hit = next((a for a in aliases if a in params), None)
        if hit is None:
            raise RuntimeError(f"could not map input '{ours}' to backend params {sorted(params)}")
        kwargs[hit] = fx[ours]
    return fn(**kwargs)


def main() -> None:
    import xlstm

    version = getattr(xlstm, "__version__", "unknown")
    print(f"xlstm version: {version}")

    qualname, fn = find_backend()
    fx = build_fixture()
    out: dict = {"xlstm_version": version, "fixture_spec": FIXTURE}

    if fn is None:
        print("!! no parallel_stabilized backend found — dumping package tree for manual inspection")
        out["submodules"] = [info.name for info in pkgutil.walk_packages(xlstm.__path__, prefix="xlstm.")]
    else:
        print(f"backend: {qualname}")
        print(f"signature: {inspect.signature(fn)}")
        h = call_backend(fn, fx).detach().to(torch.float64)
        out["backend"] = qualname
        out["signature"] = str(inspect.signature(fn))
        out["inputs"] = {k: t.tolist() for k, t in fx.items()}
        out["output_h"] = h.tolist()
        out["output_shape"] = list(h.shape)
        print(f"output shape: {tuple(h.shape)}  finite={torch.isfinite(h).all().item()}")

    with open("xlstm_probe.json", "w") as fh:
        json.dump(out, fh)
    print("wrote xlstm_probe.json — send this back for the parity test")


if __name__ == "__main__":
    main()
