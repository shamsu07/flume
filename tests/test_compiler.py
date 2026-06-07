from flume.compiler import ContextPackCompiler
from flume.models import DocumentChunk, PackCreateRequest


def _request(version: str = "1", tenant_id: str = "tenant-a") -> PackCreateRequest:
    return PackCreateRequest(
        tenant_id=tenant_id,
        model_id="model-a",
        tokenizer_id="tokenizer-a",
        chunks=[
            DocumentChunk(doc_id="b", chunk_id="2", version=version, text="Second chunk"),
            DocumentChunk(doc_id="a", chunk_id="1", version=version, text="First chunk"),
        ],
    )


def test_pack_compilation_is_deterministic() -> None:
    compiler = ContextPackCompiler()
    first = compiler.compile(_request())
    second = compiler.compile(_request())

    assert first.pack_id == second.pack_id
    assert first.document_hash == second.document_hash
    assert first.token_hash == second.token_hash
    assert first.compiled_prefix == second.compiled_prefix


def test_stable_ordering_sorts_chunks() -> None:
    compiler = ContextPackCompiler()
    pack = compiler.compile(_request())

    first_pos = pack.compiled_prefix.index("doc_id='a'")
    second_pos = pack.compiled_prefix.index("doc_id='b'")
    assert first_pos < second_pos


def test_document_version_invalidates_pack() -> None:
    compiler = ContextPackCompiler()
    first = compiler.compile(_request(version="1"))
    second = compiler.compile(_request(version="2"))

    assert first.pack_id != second.pack_id
    assert first.document_hash != second.document_hash


def test_tenant_isolation_invalidates_pack_id() -> None:
    compiler = ContextPackCompiler()
    first = compiler.compile(_request(tenant_id="tenant-a"))
    second = compiler.compile(_request(tenant_id="tenant-b"))

    assert first.pack_id != second.pack_id
    assert first.document_hash != second.document_hash
