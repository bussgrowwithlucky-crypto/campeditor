from app.models import Title, Transcript, TranscriptWord
from app.rendering import _build_ass, _title_ass_text
from app.title_generation import (
    enforce_layout_rules,
    fallback_title,
    limit_title_highlights,
    _needs_transcript_specific_override,
)


def test_full_name_counts_as_one_highlight() -> None:
    title = limit_title_highlights(
        Title(
            line1="Steve Jobs Changed Everything",
            line2="This Mistake Kills Companies",
            highlight_words=["Steve", "Kills Companies"],
        )
    )

    assert title.highlight_words == ["Steve Jobs", "Kills"]


def test_title_renderer_highlights_only_one_phrase_per_line() -> None:
    text = _title_ass_text(
        Title(
            line1="Elon Musk Built AI",
            line2="AI Changed Everything",
            highlight_words=["Elon Musk", "AI"],
        )
    )

    line1, line2 = text.split(r"\N")
    assert line1.count(r"\c&H1111CD&") == 1
    assert line2.count(r"\c&H1111CD&") == 1
    assert r"\fnRubik" not in text
    assert r"\fnInter" in text


def test_title_lines_do_not_keep_full_stops() -> None:
    title = limit_title_highlights(
        Title(
            line1="Steve Jobs' Secret Technique.",
            line2="A Trillion-Dollar Company Revealed.",
            highlight_words=["Steve Jobs", "Trillion-Dollar"],
        )
    )

    assert title.line1 == "Steve Jobs' Secret Technique"
    assert title.line2 == "A Trillion-Dollar Company Revealed"
    assert "." not in f"{title.line1}{title.line2}"


def test_fallback_title_is_viral_not_transcript_first_words() -> None:
    title = fallback_title(
        "Blindly listening to customers will kill your company. "
        "When Henry Ford built the car, he didn't ask people what they wanted "
        "because they would have asked for faster horses. "
        "When Steve Jobs built the iPhone, he didn't ask customers if they wanted the iPhone."
    )

    # 7 words total: line1 must carry at least 5, leaving a 2-word line2 —
    # which per the layout rules cannot hold a highlight.
    assert title.line1 == "Henry Ford's Secret Rule For"
    assert title.line2 == "Ignoring Customers"
    assert title.highlight_words == ["Henry Ford"]
    assert "Blindly listening" not in f"{title.line1} {title.line2}"


def test_caption_shadow_is_50_percent() -> None:
    ass = _build_ass(
        Transcript(words=[TranscriptWord(word="hello", start=0, end=1)]),
        Title(line1="Test Title", line2="", highlight_words=[]),
        2,
    )

    assert "Style: Caption,Inter,66,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000" in ass


def test_collapse_short_title_to_one_line() -> None:
    # 4 words total split across two lines → must collapse to one.
    title = enforce_layout_rules(
        Title(
            line1="Steve Jobs Rule",
            line2="For Founders",
            highlight_words=["Steve Jobs"],
        )
    )
    assert title.line1 == "Steve Jobs Rule For Founders"
    assert title.line2 == ""

    # 5 words total split across two lines → still collapses.
    title = enforce_layout_rules(
        Title(
            line1="Jobs Mistake Was",
            line2="Hiding Here",
            highlight_words=["Jobs"],
        )
    )
    assert title.line1 == "Jobs Mistake Was Hiding Here"
    assert title.line2 == ""

    # 6 words total → stays two lines, but line1 must carry at least 5 words.
    title = enforce_layout_rules(
        Title(
            line1="His Secret Rule",
            line2="For Building Apple",
            highlight_words=["His", "Apple"],
        )
    )
    assert title.line1 == "His Secret Rule For Building"
    assert title.line2 == "Apple"
    # 1-word line2 cannot hold a highlight; the line2-bound one is dropped.
    assert "Apple" not in title.highlight_words

    # 3 words total already on a single line → stays.
    title = enforce_layout_rules(
        Title(line1="The Big Secret", line2="", highlight_words=["Secret"])
    )
    assert title.line1 == "The Big Secret"
    assert title.line2 == ""


def test_no_highlight_on_two_word_line2() -> None:
    title = enforce_layout_rules(
        Title(
            line1="Jeff Bezos Built Amazon",
            line2="In Years",
            highlight_words=["Jeff Bezos", "Years"],
        )
    )
    # The line2-bound highlight must be dropped; line1 highlight survives.
    assert "Jeff Bezos" in title.highlight_words
    assert "Years" not in title.highlight_words

    # Downstream highlight picker should now select a line1 phrase.
    finalized = limit_title_highlights(title)
    line1_rendered, _ = _title_ass_text(finalized).split(r"\N")
    assert r"\c&H1111CD&" in line1_rendered


def test_highlights_survive_normal_two_line_title() -> None:
    # line1 has 5 words and line2 has 4+ words → no re-split, no stripping.
    title = enforce_layout_rules(
        Title(
            line1="Mark Cuban Just Leaked The",
            line2="AI Rule For Founders",
            highlight_words=["Mark Cuban", "AI"],
        )
    )
    assert title.highlight_words == ["Mark Cuban", "AI"]
    assert title.line2 == "AI Rule For Founders"


def test_no_highlight_on_three_word_line2() -> None:
    # line2 with exactly 3 words also may not be colored (user rule: 1-3 words).
    title = enforce_layout_rules(
        Title(
            line1="Jeff Bezos Works Twelve Hours",
            line2="Every Single Day",
            highlight_words=["Jeff Bezos", "Day"],
        )
    )
    assert title.line2 == "Every Single Day"
    assert "Day" not in title.highlight_words
    assert "Jeff Bezos" in title.highlight_words


def test_generic_money_highlight_collapses_to_one_word() -> None:
    title = limit_title_highlights(
        Title(
            line1="He Built A $100 Billion Company",
            line2="Without Any Funding",
            highlight_words=["$100 Billion", "Any Funding"],
        )
    )

    assert "$100 Billion" not in title.highlight_words
    assert "$100" in title.highlight_words
    assert all(len(highlight.split()) == 1 for highlight in title.highlight_words)


def test_embedded_full_name_stays_together_but_generic_span_collapses() -> None:
    title = limit_title_highlights(
        Title(
            line1="Steve Jobs Secret Rule Changed",
            line2="Built A Billion Dollar Company",
            highlight_words=["Steve Jobs Secret", "Billion Dollar Company"],
        )
    )

    assert title.highlight_words == ["Steve Jobs", "Billion"]


def test_layout_rules_apply_to_fallback_titles() -> None:
    # The "AI Secret / Nobody Is Saying" fallback is 5 words → should
    # collapse to one line.
    title = fallback_title("Some random transcript without names or numbers")
    assert len(f"{title.line1} {title.line2}".split()) != 5 or title.line2 == ""


def test_fallback_title_uses_promotion_story_not_generic_number_rule() -> None:
    title = fallback_title(
        "I have a story of a 23 year old who vibe coded automations and AI apps. "
        "The CEO could not believe it. They immediately promoted them twice and "
        "put them in an AI committee."
    )

    assert title.line1 == "This 23-Year-Old Got Promoted Twice"
    assert title.line2 == "After One AI Move"
    assert "Nobody Talks About" not in f"{title.line1} {title.line2}"


def test_generic_number_rule_title_is_rejected_for_promotion_story() -> None:
    transcript = (
        "This 23 year old built AI automations, changed the workflow, "
        "and got promoted twice in the same company."
    )
    title = Title(
        line1="The 23 Rule",
        line2="Nobody Talks About",
        highlight_words=["23", "Rule"],
    )

    assert _needs_transcript_specific_override(title, transcript)


def test_named_promotion_fallback_keeps_full_name() -> None:
    title = fallback_title(
        "Maya Patel was 23 when she used AI automations at work and got promoted twice."
    )

    assert title.line1 == "Maya Patel Got Promoted Twice"
    assert title.line2 == "At 23 With One AI Move"
    assert "Maya Patel" in title.highlight_words
