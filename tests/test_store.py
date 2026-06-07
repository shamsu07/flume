from flume.compiler import ContextPackCompiler
from flume.models import DocumentChunk, PackCreateRequest
from flume.store import FlumeStore


def test_store_round_trips_pack(tmp_path) -> None:
    store = FlumeStore(f"sqlite:///{tmp_path / 'flume.db'}")
    store.init_schema()
    pack = ContextPackCompiler().compile(
        PackCreateRequest(
            tenant_id="demo",
            model_id="model",
            tokenizer_id="tokenizer",
            chunks=[DocumentChunk(doc_id="doc", text="hello")],
        )
    )

    store.save_pack(pack)
    loaded = store.get_pack(pack.pack_id)

    assert loaded is not None
    assert loaded.pack_id == pack.pack_id
    assert loaded.compiled_prefix == pack.compiled_prefix
    assert len(store.list_packs()) == 1


def test_store_tracks_routes(tmp_path) -> None:
    store = FlumeStore(f"sqlite:///{tmp_path / 'flume.db'}")
    store.init_schema()

    store.save_route("pack-a", "http://worker-a", hit=True)
    store.save_route("pack-a", "http://worker-a", hit=True)
    route = store.get_route("pack-a")

    assert route is not None
    assert route.worker_url == "http://worker-a"
    assert route.affinity_hits == 2
