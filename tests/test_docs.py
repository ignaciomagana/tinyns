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


def test_benchmark_readme_post_cleanup_validation_summary() -> None:
    """Lock in the concise post-cleanup validation summary without every number."""
    text = (Path(__file__).resolve().parents[1] / "benchmarks" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "Post-cleanup overnight validation" in text
    assert "B32 cached block" in text
    assert "Live-cov" in text
    assert "do not promote" in text
