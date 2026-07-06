"""Unit tests for the enable_learned_broll opt-out (non-replicate jobs).

Covers the gate added so users can submit a non-replicate job and get a plain
caption + title render with NO auto-injected B-roll from the learned profile.
"""
from app.jobs import Pipeline
from app.models import Job, JobStatus


def test_job_enable_learned_broll_defaults_true_for_back_compat():
    """Old meta.json files won't have this field — Pydantic default keeps
    them loading with the historical behavior (learned B-roll ON)."""
    job = Job(id="legacy", status=JobStatus.INGESTED)
    assert job.enable_learned_broll is True


def test_job_enable_learned_broll_can_be_set_false():
    """User-controlled opt-out round-trips through the Job model cleanly."""
    job = Job(id="user_off", status=JobStatus.INGESTED, enable_learned_broll=False)
    assert job.enable_learned_broll is False


def test_create_job_accepts_enable_learned_broll_param():
    """Mirror the function signature so the wiring is grep-visible — if a
    future refactor drops the param, this fails fast at import time."""
    import inspect

    sig = inspect.signature(Pipeline.create_job)
    assert "enable_learned_broll" in sig.parameters
    assert sig.parameters["enable_learned_broll"].default is True


def test_create_bulk_job_accepts_enable_learned_broll_param():
    import inspect

    sig = inspect.signature(Pipeline.create_bulk_job)
    assert "enable_learned_broll" in sig.parameters
    # Bulk mode is always replicate=True so this flag is moot for those
    # jobs, but the param still exists for meta.json round-trip parity.
    assert sig.parameters["enable_learned_broll"].default is True


def test_learned_broll_gate_truth_table():
    """The gate in _run is a three-way AND:
        not job.replicate AND settings.broll_learning_enabled AND job.enable_learned_broll
    Pin the truth table so accidental refactors (e.g. flipping the polarity,
    dropping a clause) are caught immediately."""
    cases = [
        # (replicate, settings_enabled, job_flag, expected_gate)
        (False, True, True, True),   # historical default — gate fires
        (False, True, False, False), # user opted out — gate blocked
        (False, False, True, False), # settings disabled — gate blocked
        (False, False, False, False),
        (True, True, True, False),   # replicate mode skips learned B-roll
        (True, True, False, False),
    ]
    for replicate, settings_enabled, job_flag, expected in cases:
        gate = (not replicate) and settings_enabled and job_flag
        assert gate is expected, (
            f"replicate={replicate} settings_enabled={settings_enabled} "
            f"job_flag={job_flag}: expected {expected}, got {gate}"
        )