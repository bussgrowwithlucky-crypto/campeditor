import json

from app.clip_selection import _window_score, fallback_viral_clip
from app.config import Settings
from app.models import Transcript, TranscriptSegment, TranscriptWord
from app.viral_profile import ViralProfile, load_viral_profile


def test_learned_profile_changes_fallback_clip_score(tmp_path) -> None:
    (tmp_path / "viral_profile.json").write_text(
        json.dumps(
            {
                "hook_terms": {"amazon": 8, "trillion": 8},
                "phrases": {"work life harmony": 10, "built amazon": 10},
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        groq_api_key="",
        llm_api_key="",
        data_dir=tmp_path,
    )
    settings.auto_clip_target_seconds = 5
    settings.auto_clip_min_seconds = 3
    settings.auto_clip_max_seconds = 6
    transcript = Transcript(
        segments=[
            TranscriptSegment(start=0, end=4, text="This part is ordinary context"),
            TranscriptSegment(
                start=6,
                end=10,
                text="Work life harmony built Amazon into a trillion dollar company",
            ),
        ]
    )

    selection = fallback_viral_clip(transcript, 12, settings)

    assert selection.start == 6


def test_learned_opening_profile_is_loaded_from_records(tmp_path) -> None:
    (tmp_path / "viral_profile.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "opening": "Work-life balance is bullshit. Jeff Bezos built Amazon from sacrifice."
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    profile = load_viral_profile(Settings(groq_api_key="", llm_api_key="", data_dir=tmp_path))

    assert "bullshit" in profile.opening_terms
    assert "jeff bezos" in profile.opening_phrases


def test_opening_hook_scores_above_buried_hook() -> None:
    profile = ViralProfile(
        hook_terms={"trillion": 8, "amazon": 8, "bullshit": 6},
        phrases={"trillion dollar company": 12},
        opening_terms={"bullshit": 12, "amazon": 8},
        opening_phrases={"work life harmony": 14},
    )
    transcript = Transcript(
        words=[
            TranscriptWord(word="ordinary", start=0, end=0.5),
            TranscriptWord(word="setup", start=0.6, end=1.0),
            TranscriptWord(word="keeps", start=1.1, end=1.5),
            TranscriptWord(word="going", start=1.6, end=2.0),
            TranscriptWord(word="trillion", start=3.8, end=4.1),
            TranscriptWord(word="dollar", start=4.2, end=4.5),
            TranscriptWord(word="company", start=4.6, end=4.9),
            TranscriptWord(word="bullshit", start=8.0, end=8.4),
            TranscriptWord(word="work-life", start=8.5, end=8.9),
            TranscriptWord(word="harmony", start=9.0, end=9.3),
            TranscriptWord(word="built", start=9.4, end=9.7),
            TranscriptWord(word="Amazon", start=9.8, end=10.1),
        ]
    )

    assert _window_score(transcript, 8.0, 13.0, profile) > _window_score(transcript, 0.0, 5.0, profile)
