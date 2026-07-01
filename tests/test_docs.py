from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def test_readme_support_tiers_keep_key_paths() -> None:
    """Lock in the lightweight support matrix without asserting full paragraphs."""
    text = README.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "Recommended fast path" in text
    assert 'sample="rwalk"' in text
    assert 'kernel="jax"' in text
    assert "jax_block_size=32" in text

    assert "Reference baseline" in text
    assert 'sample="prior"' in text
    assert "dynesty" in lowered
    assert "slice/random-slice samplers were removed" in lowered
    removed_slice = 'sample="' + 'slice"'
    removed_rslice = 'sample="' + 'rslice"'
    removed_steps = 'slice' + '_steps'
    assert removed_slice not in text
    assert removed_rslice not in text
    assert removed_steps not in text

    assert "experimental" in lowered
    assert 'rwalk_proposal="live-cov"' in text
    assert 'bound="multi"' in text


def test_release_checklist_matches_public_surface_cleanup() -> None:
    """Keep the release checklist aligned with the current sampler support story."""
    text = (ROOT / "RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    lowered = text.lower()

    assert "gaussian_2d_rwalk_jax_block.py" in text
    assert "no ellipsoidal bounding" not in lowered
    assert "ellipsoidal" in lowered
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


def test_benchmark_readme_post_cleanup_validation_summary() -> None:
    """Lock in the concise post-cleanup validation summary without every number."""
    text = (Path(__file__).resolve().parents[1] / "benchmarks" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "Post-cleanup overnight validation" in text
    assert "B32 cached block" in text
    assert "Live-cov" in text
    assert "do not promote" in text
    assert "jax_block_size=64" in text
    assert "jax_block_size=128" in text
    assert "B32 remains the recommended" in text
    assert "B64 is the default" not in text
    assert "B128 is the default" not in text


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
