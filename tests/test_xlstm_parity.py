"""Tier-2 parity: our vendored mLSTM must match the official ``xlstm`` reference backend.

The fixture ``tests/fixtures/xlstm_probe.json`` is produced by
``scripts/xlstm/probe_official_xlstm.py`` on a box with the official ``xlstm`` package installed
(validated against xlstm 2.0.5, backend ``parallel_stabilized_simple``). It holds a seeded set of
inputs and the reference output ``h``; here we run our operator on the same inputs and require
bit-level agreement. Regenerate the fixture (and bump the version note) if ``xlstm`` changes its
backend math.
"""

import json
from pathlib import Path

import pytest
import torch

from hss.model.xlstm import mlstm_parallel


FIXTURE = Path(__file__).parent / "fixtures" / "xlstm_probe.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="missing probe fixture (run scripts/xlstm/probe_official_xlstm.py)")
def test_matches_official_parallel_stabilized():
    data = json.loads(FIXTURE.read_text())
    assert "parallel_stabilized" in data["backend"], f"unexpected backend: {data['backend']}"

    inputs = data["inputs"]
    args = (torch.tensor(inputs[name], dtype=torch.float64) for name in ("q", "k", "v", "igate_preact", "fgate_preact"))
    ours = mlstm_parallel(*args, eps=1e-6)  # official default eps=1e-6, stabilize_rowwise=True
    ref = torch.tensor(data["output_h"], dtype=torch.float64)

    assert ours.shape == ref.shape
    max_diff = (ours - ref).abs().max().item()
    assert max_diff < 1e-12, f"parity with {data['backend']} (xlstm {data['xlstm_version']}) broke: {max_diff}"
