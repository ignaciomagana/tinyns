from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_documents_the_single_sampler_and_its_kwargs() -> None:
    """Lock in the public keyword list without asserting full paragraphs."""
    text = README.read_text(encoding="utf-8")
    lowered = text.lower()

    for kwarg in (
        "nlive=500",
        "walks=None",
        "replacement_chains=1",
        "block_size=32",
        "cluster_swap=None",
    ):
        assert kwarg in text
    assert "walks=max(25, 6 * ndim)" in text
    assert "dynesty" in lowered
    for removed in (
        'sample="',
        'kernel="',
        "rwalk_proposal",
        "step_scale",
        "rwalk_target_accept",
        "min_accepts",
        "max_attempts",
        "batch_size=",
        "vectorized",
        "replacement_chain_schedule",
        "jax_block_size",
        'bound="',
        "slice" + "_steps",
    ):
        assert removed not in text, removed
    assert not re.search(r"\bisotropic", text)


def test_release_checklist_matches_public_surface_cleanup() -> None:
    """Keep the release checklist aligned with the current sampler support story."""
    text = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    lowered = text.lower()

    assert "gaussian_2d_rwalk_jax_block.py" in text
    assert "experimental" in lowered
    assert "make overnight-b32" in text
    assert "B32 overnight remains the release gate" in text
    assert "B64/B128 are optional diagnostics" in text
    assert "not part of release gating" in text


def test_changelog_alpha_release_notes_are_conservative() -> None:
    """Lock alpha notes to the narrowed public surface and caveat language."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    lowered = text.lower()

    assert "## v0.1.0-alpha" in text
    assert 'sample="rwalk"' in text
    assert 'sample="prior"' in text
    assert "jax_block_size=32" in text
    assert "B32 remains the default recommendation" in text
    assert "B64/B128" in text
    assert "optional" in lowered
    assert "experimental" in lowered
    assert "10D GW-like benchmark is a stress test" in text
    assert "not a production GW parameter-estimation pipeline" in text
    assert '`sample="bound"` public sampler paths' in text
    assert 'sample="slice"' not in text
    assert 'sample="rslice"' not in text


def test_benchmark_readme_documents_block_size_validation() -> None:
    """Keep the benchmark docs on the default sampler's own knobs."""
    text = (Path(__file__).resolve().parents[1] / "benchmarks" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "--block-sizes" in text
    assert "live_cov_B1" in text
    assert "make overnight-b32" in text
    for removed in ("--kernel", "--samplers", "jax_block_size", "--step-scale"):
        assert removed not in text, removed
    assert not re.search(r"\bisotropic", text)


def test_benchmark_readme_gw_like_stress_target_guidance() -> None:
    """Keep the GW-like stress-target docs focused on diagnostics, not defaults."""
    text = (Path(__file__).resolve().parents[1] / "benchmarks" / "README.md").read_text(
        encoding="utf-8"
    )
    lowered = text.lower()

    assert "10D GW-like" in text
    assert "insertion-rank" in text
    assert "--walks 160" in text
    assert "not as a new default configuration" in lowered
    assert "not part of the release gate" in lowered
