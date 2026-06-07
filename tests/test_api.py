from fastapi.testclient import TestClient

from flume.api import create_app
from flume.config import Settings


def test_health_endpoint(tmp_path) -> None:
    app = create_app(
        Settings(
            database_url=f"sqlite:///{tmp_path / 'flume.db'}",
            vllm_workers=["http://localhost:8000"],
        )
    )
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_create_and_get_pack(tmp_path) -> None:
    app = create_app(
        Settings(
            database_url=f"sqlite:///{tmp_path / 'flume.db'}",
            vllm_workers=["http://localhost:8000"],
        )
    )
    client = TestClient(app)

    response = client.post(
        "/packs",
        json={
            "tenant_id": "demo",
            "model_id": "model",
            "tokenizer_id": "tokenizer",
            "chunks": [{"doc_id": "doc", "text": "hello"}],
        },
    )

    assert response.status_code == 200
    pack = response.json()
    loaded = client.get(f"/packs/{pack['pack_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["pack_id"] == pack["pack_id"]
