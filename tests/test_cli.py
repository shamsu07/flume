from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import flume.cli as cli


class FakeClient(AbstractContextManager["FakeClient"]):
    calls: list[tuple[str, Any]] = []

    def __init__(self, api: str, *, tenant_id: str) -> None:
        self.api = api
        self.tenant_id = tenant_id

    def __exit__(self, *args: object) -> None:
        return None

    def register_pack(self, request: object) -> SimpleNamespace:
        self.calls.append(("register", request))
        return SimpleNamespace(model_dump_json=lambda **_: '{"pack_id":"pack_1"}')

    def list_packs(self) -> SimpleNamespace:
        item = SimpleNamespace(
            pack_id="pack_1",
            model_id="model",
            token_count=42,
            created_at="2026-07-29T00:00:00Z",
        )
        return SimpleNamespace(items=[item])

    def warm_pack(self, pack_id: str) -> SimpleNamespace:
        self.calls.append(("warm", pack_id))
        return SimpleNamespace(model_dump_json=lambda **_: '{"warmed":true}')

    def complete(self, request: object) -> SimpleNamespace:
        self.calls.append(("complete", request))
        choice = SimpleNamespace(text="answer")
        return SimpleNamespace(
            choices=[choice],
            model_dump_json=lambda **_: '{"choices":[{"text":"answer"}]}',
        )

    def stats(self) -> SimpleNamespace:
        return SimpleNamespace(model_dump_json=lambda **_: '{"packs":1}')


def test_serve_constructs_authoritative_settings(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_create_app(settings: object) -> object:
        captured["settings"] = settings
        return "application"

    def fake_run(application: object, *, host: str, port: int) -> None:
        captured.update(application=application, host=host, port=port)

    monkeypatch.setattr("flume.api.create_app", fake_create_app)
    monkeypatch.setattr(cli.uvicorn, "run", fake_run)

    cli.serve(
        host="0.0.0.0",
        port=9000,
        database_url="sqlite:///test.db",
        vllm_worker=["http://worker/"],
        model="model",
        tokenizer="tokenizer",
        allow_remote_tokenizer=True,
    )

    settings = captured["settings"]
    assert settings.vllm_workers == ["http://worker"]
    assert settings.model_id == "model"
    assert captured == {
        "settings": settings,
        "application": "application",
        "host": "0.0.0.0",
        "port": 9000,
    }


def test_cli_pack_and_completion_commands(monkeypatch, tmp_path: Path, capsys) -> None:
    FakeClient.calls.clear()
    source = tmp_path / "context.txt"
    source.write_text("context", encoding="utf-8")
    monkeypatch.setattr(cli, "FlumeClient", FakeClient)

    cli.create_pack(
        files=[source],
        tenant="tenant-a",
        template_id="template-a",
        api="http://flume",
    )
    cli.list_packs(tenant="tenant-a", api="http://flume")
    cli.warm(pack_id="pack_1", tenant="tenant-a", api="http://flume")
    cli.ask(
        pack_id="pack_1",
        prompt="question",
        tenant="tenant-a",
        api="http://flume",
        max_tokens=7,
    )
    cli.stats(tenant="tenant-a", api="http://flume")

    register = FakeClient.calls[0][1]
    assert register.template_id == "template-a"
    assert register.chunks[0].doc_id == "context.txt"
    assert FakeClient.calls[1] == ("warm", "pack_1")
    completion = FakeClient.calls[2][1]
    assert completion.pack_id == "pack_1"
    assert completion.prompt == "question"
    assert completion.max_tokens == 7
    assert "answer" in capsys.readouterr().out


def test_main_invokes_typer_application(monkeypatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(cli, "app", lambda: called.append(True))

    cli.main()

    assert called == [True]
