from __future__ import annotations

import numpy as np

from tinyns import NestedSamplerResult


def test_result_asdict_returns_expected_fields() -> None:
    result = NestedSamplerResult(
        samples=np.zeros((2, 3)),
        logl=np.array([-1.0, -0.5]),
        logwt=np.array([-2.0, -1.5]),
        logz=-0.1,
        logzerr=0.01,
        niter=2,
        metadata={"status": "scaffold"},
    )

    data = result.asdict()

    assert set(data) == {
        "samples",
        "logl",
        "logwt",
        "logz",
        "logzerr",
        "niter",
        "metadata",
    }
    assert data["metadata"] == {"status": "scaffold"}
    assert data["metadata"] is not result.metadata


def test_result_nsamples_uses_first_sample_axis() -> None:
    result = NestedSamplerResult(
        samples=np.zeros((4, 2)),
        logl=np.zeros(4),
        logwt=np.zeros(4),
        logz=0.0,
        logzerr=0.0,
        niter=4,
    )

    assert result.nsamples == 4
