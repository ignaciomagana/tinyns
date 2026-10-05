import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "tools" / "ab_bitwise.py"
_spec = importlib.util.spec_from_file_location("ab_bitwise", _PATH)
ab_bitwise = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab_bitwise)


def _save(path, **arrays):
    np.savez(path, **arrays)
    return path


def test_compare_identical(tmp_path):
    x = np.array([1.0, np.nan, 3.0])
    a = _save(tmp_path / "a.npz", **{"c/logl": x, "c/ncall": np.asarray(7)})
    b = _save(tmp_path / "b.npz", **{"c/logl": x.copy(), "c/ncall": np.asarray(7)})
    keys, bad, notes = ab_bitwise.compare(a, b)
    assert keys == ["c/logl", "c/ncall"]
    assert bad == [] and notes == []


def test_compare_flags_value_shape_and_missing(tmp_path):
    a = _save(
        tmp_path / "a.npz",
        **{"c/logz": np.asarray(0.1), "c/logl": np.zeros(3), "c/samples_u": np.ones(2)},
    )
    b = _save(
        tmp_path / "b.npz",
        **{"c/logz": np.asarray(0.1 + 1e-15), "c/logl": np.zeros(4)},
    )
    _, bad, notes = ab_bitwise.compare(a, b)
    assert {k for k, _ in bad} == {"c/logz", "c/logl", "c/samples_u"}
    assert notes == []


def test_compare_metadata_presence_is_a_note(tmp_path):
    a = _save(tmp_path / "a.npz", **{"c/logz": np.asarray(0.0)})
    b = _save(
        tmp_path / "b.npz",
        **{"c/logz": np.asarray(0.0), "c/meta_cluster_swap_accepts": np.asarray(3)},
    )
    _, bad, notes = ab_bitwise.compare(a, b)
    assert bad == []
    assert [k for k, _ in notes] == ["c/meta_cluster_swap_accepts"]


def test_cases_name_known_targets():
    targets = ab_bitwise._targets()
    for name, (target, nlive, opt) in ab_bitwise.CASES.items():
        assert target in targets, name
        assert nlive > 0, name
        assert set(opt) <= {"chains", "block", "maxiter", "resume_at", "walks"}, name
