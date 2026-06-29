from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_support_tiers_keep_key_paths() -> None:
    """Lock in the lightweight support matrix without asserting full paragraphs."""
    text = README.read_text(encoding="utf-8")
    lowered = text.lower()

    assert "Recommended fast path" in text
    assert 'sample="rwalk"' in text
    assert 'kernel="jax"' in text
    assert "jax_block_size=32" in text

    assert "Reference-only" in text or "frozen" in lowered
    assert 'sample="slice"' in text
    assert 'sample="rslice"' in text

    assert "experimental" in lowered
    assert 'rwalk_proposal="live-cov"' in text
    assert 'bound="multi"' in text
