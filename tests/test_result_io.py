"""Result files and weight statistics: ``save_npz`` writes the exact path
(through a link), ``load_npz`` keeps the saved dtypes, and the weights, ESS,
information and resampling are computed in NumPy float64 (without compiling
anything), with the resampled set in a random order."""

from __future__ import annotations

import io
import logging
import os
import re
import stat
import subprocess
import sys
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tinyns import NestedSamplingResult
from tinyns.math import normalize_log_weights
from tinyns.result import _RESULT_NPZ_ARRAYS


def make_result(n: int, offset: float = 0.0) -> NestedSamplingResult:
    """A float64 result of ``n`` samples by hand, at a log-likelihood offset:
    equal prior volumes, so the weights rise with the likelihood (death
    order)."""
    u = np.linspace(0.05, 0.95, 2 * n).reshape(n, 2)
    logl = offset + np.linspace(-5.0, 0.0, n)
    logwt = logl - np.log(n)
    return NestedSamplingResult(
        samples_u=u,
        samples=1000.0 + 1e-3 * u,  # values a float32 cannot hold
        logl=logl,
        logwt=logwt,
        logl_birth=np.full(n, -np.inf),
        nlive_i=np.full(n, n, np.int64),
        labels=np.zeros(n, np.int32),
        logz=float(np.logaddexp.reduce(logwt)),
        logzerr=0.1,
        ncall=10 * n,
        niter=n // 2,
        nlive=n // 2,
        num_delete=1,
        ndim=2,
    )


def reference_weights(logwt) -> np.ndarray:
    logwt = np.asarray(logwt, np.float64)
    return np.exp(logwt - np.logaddexp.reduce(logwt))


def test_save_npz_writes_the_exact_path(tmp_path) -> None:
    """``np.savez_compressed`` appended ``.npz`` to a path without it, so
    ``load_npz`` of the same path failed."""
    result = make_result(40)
    for name in ("result", "result.out", "result.npz"):
        result.save_npz(tmp_path / name)
        assert NestedSamplingResult.load_npz(tmp_path / name).logz == result.logz
    # No suffix appended and no temporary file left.
    assert sorted(os.listdir(tmp_path)) == ["result", "result.npz", "result.out"]
    buffer = io.BytesIO()  # a file object is written as it is
    result.save_npz(buffer)
    buffer.seek(0)
    assert NestedSamplingResult.load_npz(buffer).niter == result.niter


@pytest.mark.skipif(os.name != "posix", reason="symbolic links")
def test_save_npz_writes_through_a_symbolic_link(tmp_path) -> None:
    """As ``np.savez_compressed`` did: the atomic write must not replace the
    link with a file and leave the linked one stale."""
    shared = tmp_path / "shared"
    shared.mkdir()
    link = tmp_path / "result.npz"
    link.symlink_to(shared / "result.npz")
    make_result(40).save_npz(link)  # creates the target
    result = make_result(60)
    result.save_npz(link)  # replaces it
    assert link.is_symlink() and os.listdir(shared) == ["result.npz"]
    assert NestedSamplingResult.load_npz(shared / "result.npz").niter == result.niter


def test_a_failed_save_npz_names_the_file(tmp_path) -> None:
    """The atomic write's errors named its hidden temporary file."""
    path = tmp_path / "missing" / "result.npz"
    with pytest.raises(FileNotFoundError,
                       match=re.escape(f"cannot write {path} ")) as info:
        make_result(40).save_npz(path)
    assert ".result.npz." not in str(info.value)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_save_npz_keeps_the_mode_of_the_file_it_replaces(tmp_path) -> None:
    umask = os.umask(0o022)
    try:
        plain = tmp_path / "plain"
        plain.write_bytes(b"")
        path = tmp_path / "result.npz"
        make_result(40).save_npz(path)
        assert stat.S_IMODE(path.stat().st_mode) == stat.S_IMODE(plain.stat().st_mode)
        path.chmod(0o640)
        make_result(40).save_npz(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    finally:
        os.umask(umask)


def test_load_npz_keeps_the_saved_dtypes(tmp_path) -> None:
    """Without x64, loading turned float64 into float32 (and int64 into int32)
    through ``jnp.asarray``: a result saved under x64 lost its precision."""
    result = make_result(400)
    path = tmp_path / "result.npz"
    result.save_npz(path)
    loaded = NestedSamplingResult.load_npz(path)
    for name in _RESULT_NPZ_ARRAYS:
        saved, back = getattr(result, name), getattr(loaded, name)
        assert isinstance(back, np.ndarray), name
        assert back.dtype == saved.dtype, name
        np.testing.assert_array_equal(back, saved, name)
    assert len(np.unique(loaded.samples[:, 0])) == 400
    np.testing.assert_array_equal(loaded.weights(), result.weights())


def _python(code: str, path, x64: bool) -> None:
    """Run ``code`` (``sys.argv[1]`` is ``path``) in a new Python on the CPU,
    with x64 on or off and this test's ``tinyns``."""
    env = dict(
        os.environ,
        JAX_ENABLE_X64="1" if x64 else "0",
        JAX_PLATFORMS="cpu",
        PYTHONPATH=os.pathsep.join(p for p in sys.path if p),
    )
    subprocess.run([sys.executable, "-c", code, str(path)], env=env, check=True)


def test_a_result_saved_under_x64_loads_unchanged_without_x64(tmp_path) -> None:
    """The case behind the test above, end to end (a session runs in one x64
    mode, so each side is a new process): a run saved under x64 came back
    from ``load_npz`` without x64 as float32."""
    path = tmp_path / "result.npz"
    _python(
        "import sys, jax.numpy as jnp\n"
        "from tinyns import NestedSampler\n"
        "def loglike(x):\n"
        "    return -0.5 * jnp.sum(((x - 1000.005) / 1e-3) ** 2)\n"
        "result = NestedSampler(loglike, lambda u: 1000.0 + 1e-2 * u, 2, 50).run(0)\n"
        "assert result.samples.dtype == jnp.float64\n"
        "result.save_npz(sys.argv[1])\n",
        path, x64=True,
    )
    _python(
        "import sys, jax, numpy as np\n"
        "from tinyns import NestedSamplingResult\n"
        "assert not jax.config.jax_enable_x64\n"
        "result = NestedSamplingResult.load_npz(sys.argv[1])\n"
        "with np.load(sys.argv[1]) as saved:\n"
        "    for name in ('samples_u', 'samples', 'logl', 'logwt'):\n"
        "        assert getattr(result, name).dtype == np.float64, name\n"
        "        np.testing.assert_array_equal(getattr(result, name), saved[name])\n",
        path, x64=False,
    )


def test_weight_statistics_are_float64() -> None:
    """At |logl| ~ 1e6 the float32 normalization put the weights off by a few
    percent, even for a result whose ``logwt`` is float64."""
    result = make_result(1000, offset=-1e6)
    w = reference_weights(result.logwt)
    weights = result.weights()
    assert isinstance(weights, np.ndarray) and weights.dtype == np.float64
    np.testing.assert_allclose(weights, w, rtol=1e-9)
    np.testing.assert_allclose(np.exp(result.log_weights()), w, rtol=1e-9)
    assert result.posterior_ess() == pytest.approx(1.0 / np.sum(w * w), rel=1e-9)
    live = int(result.niter)
    assert result.live_weight_fraction() == pytest.approx(w[live:].sum(), rel=1e-9)
    entropy = -np.sum(w * np.log(w))
    assert result.posterior_weight_entropy() == pytest.approx(entropy, rel=1e-9)
    logl = np.asarray(result.logl)
    information = np.sum(w * (logl - result.logz))  # KL(w | uniform) > 0
    assert result.information() == pytest.approx(information, rel=1e-9)


def test_weights_sum_to_one_at_a_large_offset() -> None:
    """``logw - (max + log(sum))`` rounds the normalizer to an ulp of
    ``|max|``: in float64 the weights summed to 1 within only ~1e-11 at
    ``|logwt| ~ 1e6``."""
    z = np.random.default_rng(0).standard_normal(1000)
    for offset in -1e6 - 0.137 * np.arange(20):
        weights = np.exp(normalize_log_weights(offset - 0.5 * z**2))
        assert abs(float(np.sum(weights)) - 1.0) < 1e-14


def test_information_of_an_infinite_logz_is_nan_without_a_warning() -> None:
    """``logwt - logz`` warned (``inf - inf``) where JAX was silent."""
    result = make_result(40)
    result.logwt = np.full(40, -np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for logz in (-np.inf, np.inf):
            result.logz = logz
            assert np.isnan(result.information())


def test_resample_equal_never_runs_past_the_last_weight() -> None:
    """With float32 weights summing below 1, the positions past the end of the
    CDF were clamped to the last sample: 1000 equal weights at an offset of
    -1e6 gave it about 30 times its share."""
    result = make_result(1000)
    result.logwt = np.full(1000, -1e6)
    result.samples = np.arange(1000.0)[:, None]
    draws = result.resample_equal(0, n=4000)
    counts = np.bincount(draws[:, 0].astype(int), minlength=1000)
    np.testing.assert_array_equal(counts, 4)  # systematic: exactly n w_i each


def test_resample_equal_returns_a_random_order() -> None:
    """Systematic resampling picks the samples in death order (increasing
    likelihood), so a prefix or a split of the set was biased."""
    result = make_result(1000)
    result.samples = np.arange(1000.0)[:, None]
    draws = result.resample_equal(0, n=2000)[:, 0]
    assert np.any(np.diff(draws) < 0)
    w = reference_weights(result.logwt)
    sd = np.sqrt(w @ (np.arange(1000.0) - w @ np.arange(1000.0)) ** 2)
    first, second = draws[:1000].mean(), draws[1000:].mean()
    assert abs(first - second) < 0.2 * sd  # their standard error is 0.045 sd
    # The set itself is the systematic one: each sample floor or ceil(n w).
    counts = np.bincount(draws.astype(int), minlength=1000)
    assert np.all(np.abs(counts - 2000 * w) < 1.0)
    np.testing.assert_array_equal(result.resample_equal(0, n=2000)[:, 0], draws)
    assert not np.array_equal(result.resample_equal(1, n=2000)[:, 0], draws)


def test_resample_equal_needs_an_integer_n() -> None:
    """``n=5.0`` failed deep in NumPy (``AxisError`` of the permutation)."""
    result = make_result(40)
    for n in (5.0, True, np.True_, jnp.asarray(True)):  # a bool, as _check_int
        with pytest.raises(TypeError, match="n must be an integer"):
            result.resample_equal(0, n=n)
    for n in (5, np.int64(5), jnp.asarray(5)):
        assert result.resample_equal(0, n=n).shape == (5, 2)


def test_weight_statistics_compile_nothing_for_a_new_result() -> None:
    """Each new sample count cost dozens of compiled eager ops (0.7 s per lane
    of a batch in ``diagnostics()``)."""
    warm = make_result(500)
    warm.diagnostics()
    warm.resample_equal(0)
    warm.resample_equal(jax.random.key(0), n=50)

    compiled = []

    class Handler(logging.Handler):
        def emit(self, record):
            if record.getMessage().startswith("Finished XLA compilation"):
                compiled.append(record.getMessage())

    logger, handler = logging.getLogger("jax"), Handler()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        jax.jit(lambda x: x + 1.0)(jnp.zeros(377))  # the control compiles
        assert compiled, "compilations are not logged"
        compiled.clear()
        fresh = make_result(733)
        fresh.diagnostics()
        fresh.summary()
        fresh.information()
        fresh.resample_equal(1)
        fresh.resample_equal(jax.random.key(1), n=1234)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
    assert compiled == []
