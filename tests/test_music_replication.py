from pathlib import Path

from app.jobs import _music_render_options
from app.models import Job, ReferenceAnalysis, Title, Transcript
from app.rendering import render
from app.config import Settings
from app.models import ColorGrade


def test_reference_music_uses_reference_volume_and_no_loop() -> None:
    reference_music = Path("data/jobs/job/reference/music_instrumental.m4a")
    job = Job(
        id="job",
        reference=ReferenceAnalysis(
            music_path=reference_music,
            music_volume_db=-31.5,
        ),
    )

    music_path, music_volume_db, music_loop = _music_render_options(job)

    assert music_path == reference_music
    assert music_volume_db == -31.5
    assert music_loop is False


def test_uploaded_music_keeps_loop_and_default_volume() -> None:
    uploaded_music = Path("data/jobs/job/music.mp3")
    reference_music = Path("data/jobs/job/reference/music_instrumental.m4a")
    job = Job(
        id="job",
        music_path=uploaded_music,
        reference=ReferenceAnalysis(
            music_path=reference_music,
            music_volume_db=-31.5,
        ),
    )

    music_path, music_volume_db, music_loop = _music_render_options(job)

    assert music_path == uploaded_music
    # When the user uploads their own music, the reference's measured loudness
    # is still used as the TARGET so the rendered track lands at the same
    # volume as the reference. Previously this was None (defaulting to -15 dB)
    # which caused the music to be noticeably louder than the reference.
    assert music_volume_db == -31.5
    assert music_loop is False


def test_render_does_not_loop_timeline_aligned_reference_music(monkeypatch, tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    music = tmp_path / "music.m4a"
    music.write_bytes(b"music")
    output = tmp_path / "render.mp4"
    captured = {}

    class Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(command, **kwargs):
        captured["command"] = command
        output.write_bytes(b"render")
        return Result()

    monkeypatch.setattr("app.rendering.probe_duration", lambda path, settings: 5.0)
    monkeypatch.setattr("app.rendering._detect_face_focus_x", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.rendering._probe_has_audio", lambda *args, **kwargs: False)
    monkeypatch.setattr("app.rendering._music_volume_coefficient", lambda *args, **kwargs: 0.73)
    monkeypatch.setattr("app.rendering.subprocess.run", fake_run)

    render(
        source_path=source,
        output_path=output,
        start=0.0,
        end=4.0,
        transcript=Transcript(),
        title=Title(line1="A", line2="B"),
        color_grade=ColorGrade.NONE,
        settings=Settings(groq_api_key="", llm_api_key=""),
        music_path=music,
        music_volume_db=-31.5,
        music_loop=False,
    )

    command = captured["command"]
    assert "-stream_loop" not in command
    assert str(music) in command
    assert any("volume=0.730" in part for part in command)

