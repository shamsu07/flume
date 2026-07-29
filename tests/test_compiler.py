import pytest
from pydantic import ValidationError

from flume.compiler import ContextPackCompiler, DeterministicByteTokenizer, load_tokenizer
from flume.hashing import sha256_json
from flume.models import DocumentChunk, PackCreateRequest
from flume.sdk import chunks_from_files


def _compiler(
    *,
    tokenizer_id: str = "tokenizer-a",
    revision: str = "revision-a",
    implementation: str = "byte-v1",
    model_id: str | None = None,
) -> ContextPackCompiler:
    class TestTokenizer:
        fingerprint = sha256_json(
            {
                "implementation": implementation,
                "tokenizer_id": tokenizer_id,
                "revision": revision,
            }
        )

        def __init__(self) -> None:
            self.tokenizer_id = tokenizer_id
            self.revision = revision

        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

    return ContextPackCompiler(TestTokenizer(), model_id=model_id)


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
    compiler = _compiler()
    first = compiler.compile(_request())
    second = compiler.compile(_request())

    assert first.pack_id == second.pack_id
    assert first.document_hash == second.document_hash
    assert first.token_hash == second.token_hash
    assert first.compiled_prefix == second.compiled_prefix


def test_pack_compilation_matches_golden_manifest_vector() -> None:
    pack = _compiler().compile(_request())

    assert pack.pack_id == "pack_52208e1b95f0e5e8facd829c"
    assert pack.document_hash == (
        "dc8a6a687ba7b7df82636725cd75ad37ad32c8825d808940e5361d7b301b88e4"
    )
    assert pack.template_digest == (
        "9a69749673a5d7663b8771b385ee70388ce67f61e049cb7626f1d2e63c137d63"
    )
    assert pack.canonical_prefix_hash == (
        "7bb89597485016f8dbe305f21bdf93da3e43633e5d9eb74586f34bb25217255c"
    )
    assert pack.token_hash == (
        "f1c1869f36c49181e4b4fb208856a4cb63d7ed2127392406de38f4329f76f78f"
    )
    assert pack.token_count == 271


def test_stable_ordering_sorts_chunks() -> None:
    compiler = _compiler()
    pack = compiler.compile(_request())

    first_pos = pack.compiled_prefix.index('doc_id="a"')
    second_pos = pack.compiled_prefix.index('doc_id="b"')
    assert first_pos < second_pos


def test_document_version_invalidates_pack() -> None:
    compiler = _compiler()
    first = compiler.compile(_request(version="1"))
    second = compiler.compile(_request(version="2"))

    assert first.pack_id != second.pack_id
    assert first.document_hash != second.document_hash


def test_tenant_isolation_invalidates_pack_id() -> None:
    compiler = _compiler()
    first = compiler.compile(_request(tenant_id="tenant-a"))
    second = compiler.compile(_request(tenant_id="tenant-b"))

    assert first.pack_id != second.pack_id
    assert first.document_hash == second.document_hash


def test_duplicate_logical_chunk_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate chunk key"):
        PackCreateRequest(
            model_id="model",
            tokenizer_id="tokenizer",
            chunks=[
                DocumentChunk(doc_id="doc", chunk_id="1", version="a", text="first"),
                DocumentChunk(doc_id="doc", chunk_id="1", version="b", text="second"),
            ],
        )


@pytest.mark.parametrize(
    "template",
    [
        "missing",
        "{context} and {context}",
        "{context!r}",
        "{context:>20}",
        "{question}: {context}",
        "{context",
    ],
)
def test_template_requires_one_plain_context_field(template: str) -> None:
    with pytest.raises(ValidationError, match="template"):
        PackCreateRequest(
            model_id="model",
            tokenizer_id="tokenizer",
            template=template,
            chunks=[DocumentChunk(doc_id="doc", text="context")],
        )


def test_input_values_are_canonicalized_and_unknown_fields_are_rejected() -> None:
    request = PackCreateRequest(
        tenant_id="  tenant  ",
        model_id="model",
        tokenizer_id="tokenizer",
        template="Header\r\n\r\n\r\n{context}\r\n",
        chunks=[
            DocumentChunk(
                doc_id="ＤＯＣ",
                text="line  \r\n\r\n\r\nnext",
                metadata={"title": "Ｃａｆｅ\u0301  "},
            )
        ],
        tags=[" z ", "a"],
        metadata={"note": "hello  \r\n"},
    )

    assert request.tenant_id == "tenant"
    assert request.template == "Header\n\n{context}"
    assert request.chunks[0].doc_id == "DOC"
    assert request.chunks[0].text == "line\n\nnext"
    assert request.chunks[0].metadata == {"title": "Café"}
    assert request.tags == ["a", "z"]
    assert request.metadata == {"note": "hello"}

    with pytest.raises(ValidationError, match="Extra inputs"):
        PackCreateRequest.model_validate(
            {
                "model_id": "model",
                "tokenizer_id": "tokenizer",
                "chunks": [{"doc_id": "doc", "text": "text", "unknown": True}],
            }
        )


def test_unicode_newlines_metadata_and_chunk_order_are_identity_invariant() -> None:
    compiler = _compiler(tokenizer_id="tokenizer")
    first = compiler.compile(
        PackCreateRequest(
            tenant_id="tenant",
            model_id="model",
            tokenizer_id="tokenizer",
            template="Résumé\r\n\r\n\r\n{context}\r\n",
            chunks=[
                DocumentChunk(
                    doc_id="Ｂ",
                    text="second\r\n",
                    metadata={"label": "Ｃａｆｅ\u0301"},
                ),
                DocumentChunk(doc_id="a", text="first  \n"),
            ],
        )
    )
    second = compiler.compile(
        PackCreateRequest(
            tenant_id="tenant",
            model_id="model",
            tokenizer_id="tokenizer",
            template="Re\u0301sume\u0301\n\n{context}",
            chunks=[
                DocumentChunk(doc_id="a", text="first"),
                DocumentChunk(
                    doc_id="B",
                    text="second",
                    metadata={"label": "Café"},
                ),
            ],
        )
    )

    assert first.pack_id == second.pack_id
    assert first.document_hash == second.document_hash
    assert first.template_digest == second.template_digest
    assert first.canonical_prefix_hash == second.canonical_prefix_hash


def test_file_chunks_ignore_path_mtime_and_argument_order(tmp_path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_a = first_root / "a.txt"
    first_b = first_root / "b.txt"
    second_a = second_root / "a.txt"
    second_b = second_root / "b.txt"
    first_a.write_text("Café\r\n", encoding="utf-8")
    first_b.write_text("other", encoding="utf-8")
    second_a.write_text("Cafe\u0301\n", encoding="utf-8")
    second_b.write_text("other", encoding="utf-8")
    second_a.touch()
    second_b.touch()

    first = chunks_from_files([first_b, first_a])
    second = chunks_from_files([second_a, second_b])

    assert first == second
    assert [chunk.doc_id for chunk in first] == ["a.txt", "b.txt"]
    assert all("path" not in chunk.metadata for chunk in first)


def test_file_chunks_accept_explicit_stable_logical_names(tmp_path) -> None:
    source = tmp_path / "renamed-on-disk.txt"
    source.write_text("content", encoding="utf-8")

    chunks = chunks_from_files([source], logical_names={source: "manual/intro.txt"})

    assert chunks[0].doc_id == "manual/intro.txt"
    assert chunks[0].metadata == {"source_name": "manual/intro.txt"}


def test_tokenizer_identity_is_pinned_and_changes_pack_identity() -> None:
    first = _compiler(implementation="byte-v1").compile(_request())
    changed_revision = _compiler(revision="revision-b").compile(_request())
    changed_implementation = _compiler(implementation="byte-v2").compile(_request())

    assert first.pack_id != changed_revision.pack_id
    assert first.pack_id != changed_implementation.pack_id
    assert first.tokenizer_revision == "revision-a"
    assert first.tokenizer_fingerprint in first.immutable_manifest().values()


def test_runtime_authority_rejects_request_model_or_tokenizer_override() -> None:
    compiler = _compiler(model_id="model-a")

    with pytest.raises(ValueError, match="tokenizer_id"):
        compiler.compile(_request().model_copy(update={"tokenizer_id": "other"}))
    with pytest.raises(ValueError, match="model_id"):
        compiler.compile(_request().model_copy(update={"model_id": "other"}))


def test_prefix_tokens_are_preserved_when_question_is_appended() -> None:
    compiler = _compiler()
    pack = compiler.compile(_request())

    prompt_ids = compiler.completion_token_ids(pack, "  What changed?  ")

    assert prompt_ids[: pack.token_count] == list(pack.prefix_token_ids)
    assert bytes(prompt_ids[pack.token_count :]).decode() == "What changed?\n\nAnswer:\n"


def test_pack_model_rejects_tampered_prefix_tokens() -> None:
    pack = _compiler().compile(_request())
    payload = pack.model_dump()
    payload["prefix_token_ids"] = [*pack.prefix_token_ids[:-1], 0]

    with pytest.raises(ValidationError, match="token_hash"):
        pack.__class__.model_validate(payload)


def test_tokenizer_loading_failure_is_not_replaced_with_fallback(monkeypatch) -> None:
    class BrokenTokenizer:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("unavailable")

    monkeypatch.setattr("flume.compiler.HuggingFaceTokenizer", BrokenTokenizer)

    with pytest.raises(RuntimeError, match="unavailable"):
        load_tokenizer("tokenizer", "deadbeef")


def test_explicit_byte_tokenizer_remains_available_for_offline_tests() -> None:
    tokenizer = DeterministicByteTokenizer(
        tokenizer_id="tokenizer-a",
        revision="test-byte-v1",
    )
    pack = ContextPackCompiler(tokenizer).compile(_request())

    assert pack.prefix_token_ids == tuple(pack.compiled_prefix.encode())
