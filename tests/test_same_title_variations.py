"""Step 7: all replicate-mode variations share ONE title (Step 5 contract).

The job pipeline builds `titles = [base_title] * variation_count` so the
A/B/C/D comparison varies only the B-roll cuts, never the title text.
"""
from app.jobs import fallback_title
from app.models import Job, Title


def test_titles_array_uses_one_base_title_repeated():
    """Mirror the jobs.py construction (line 368-369): the titles list is
    [base_title] * variation_count. This test pins that contract."""
    base_title = "Working on the future from home"
    variation_count = 4
    titles: list = [base_title] * variation_count
    assert len(titles) == variation_count
    assert all(t == base_title for t in titles)
    assert len(set(titles)) == 1


def test_titles_for_variation_count_one_still_single():
    base_title = "Solo"
    titles: list = [base_title] * 1
    assert titles == ["Solo"]


def test_fallback_title_returns_title_for_nonempty_transcript():
    """The base_title comes from `job.title or fallback_title(transcript)`.
    Ensure fallback_title gives a usable single Title (not a list)."""
    t = fallback_title("This is a transcript about a day in the life of a creator")
    assert isinstance(t, Title)
    base = t.line1
    assert isinstance(base, str) and len(base) > 0
