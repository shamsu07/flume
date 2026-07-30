import pytest
from pydantic import ValidationError

from flume.config import Settings


def test_routing_defaults_preserve_hrw() -> None:
    settings = Settings(vllm_workers=["http://a", "http://b"])

    assert settings.routing_policy == "hrw"
    assert settings.routing_load_slack == 2
    assert settings.routing_spill_hold_ms == 2_000
    assert settings.worker_load_refresh_ms == 500
    assert settings.worker_load_stale_ms == 2_000
    assert settings.worker_capacity_weights == {"http://a": 1.0, "http://b": 1.0}
    assert settings.routing_state_max_entries == 10_000
    assert settings.routing_state_ttl_seconds == 600


def test_routing_settings_validate_weights_and_windows() -> None:
    settings = Settings(
        vllm_workers=["http://a/", "http://b"],
        worker_capacity_weights={"http://a/": 2.0},
    )
    assert settings.worker_capacity_weights == {"http://a": 2.0, "http://b": 1.0}

    with pytest.raises(ValidationError, match="unknown vLLM worker"):
        Settings(
            vllm_workers=["http://a"],
            worker_capacity_weights={"http://b": 1.0},
        )
    with pytest.raises(ValidationError, match="finite and greater than zero"):
        Settings(
            vllm_workers=["http://a"],
            worker_capacity_weights={"http://a": 0.0},
        )
    with pytest.raises(ValidationError, match="at least worker_load_refresh_ms"):
        Settings(worker_load_refresh_ms=2_001, worker_load_stale_ms=2_000)
    with pytest.raises(ValidationError, match="cannot exceed"):
        Settings(routing_spill_hold_ms=2_001, routing_state_ttl_seconds=2.0)


def test_routing_environment_names_and_worker_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUME_VLLM_WORKERS", "http://a/,http://b")
    monkeypatch.setenv("FLUME_ROUTING_POLICY", "bounded_hrw")
    monkeypatch.setenv("FLUME_ROUTING_LOAD_SLACK", "3")
    monkeypatch.setenv("FLUME_ROUTING_SPILL_HOLD_MS", "1500")
    monkeypatch.setenv("FLUME_WORKER_LOAD_REFRESH_MS", "250")
    monkeypatch.setenv("FLUME_WORKER_LOAD_STALE_MS", "1000")
    monkeypatch.setenv("FLUME_WORKER_CAPACITY_WEIGHTS", '{"http://a": 2.0}')
    monkeypatch.setenv("FLUME_ROUTING_STATE_MAX_ENTRIES", "123")
    monkeypatch.setenv("FLUME_ROUTING_STATE_TTL_SECONDS", "30")

    settings = Settings(_env_file=None)

    assert settings.vllm_workers == ["http://a", "http://b"]
    assert settings.routing_policy == "bounded_hrw"
    assert settings.routing_load_slack == 3
    assert settings.routing_spill_hold_ms == 1_500
    assert settings.worker_load_refresh_ms == 250
    assert settings.worker_load_stale_ms == 1_000
    assert settings.worker_capacity_weights == {"http://a": 2.0, "http://b": 1.0}
    assert settings.routing_state_max_entries == 123
    assert settings.routing_state_ttl_seconds == 30


def test_vllm_workers_accept_json_array_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "FLUME_VLLM_WORKERS",
        '[" http://a/ ", "http://b/"]',
    )

    settings = Settings(_env_file=None)

    assert settings.vllm_workers == ["http://a", "http://b"]
