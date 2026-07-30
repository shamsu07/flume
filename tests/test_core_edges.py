from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import flume.compiler as compiler_module
from flume.compiler import ContextPackCompiler, DeterministicByteTokenizer, HuggingFaceTokenizer
from flume.config import Settings
from flume.hashing import canonical_identifier, canonical_json_value, sha256_token_ids
from flume.models import CompletionRequest, DocumentChunk, OrderPolicy, PackCreateRequest


def request(*, order_policy: OrderPolicy = OrderPolicy.stable) -> PackCreateRequest:
    return PackCreateRequest(
        tenant_id="tenant",
        model_id="model",
        tokenizer_id="tokenizer",
        chunks=[
            DocumentChunk(doc_id="z", text="last"),
            DocumentChunk(doc_id="a", text="first"),
        ],
        order_policy=order_policy,
    )


def byte_compiler() -> ContextPackCompiler:
    return ContextPackCompiler(
        DeterministicByteTokenizer(tokenizer_id="tokenizer", revision="revision"),
        model_id="model",
    )


def test_hugging_face_tokenizer_uses_pinned_resolved_revision(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class FakeTokenizer:
        _commit_hash = None
        init_kwargs = {"_commit_hash": "resolved-commit"}
        backend_tokenizer = SimpleNamespace(to_str=lambda: '{"backend":"stable"}')

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            seen["special"] = add_special_tokens
            return [len(text)]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id: str, **kwargs: object) -> FakeTokenizer:
            seen.update(tokenizer_id=tokenizer_id, **kwargs)
            return FakeTokenizer()

    monkeypatch.setattr("transformers.AutoTokenizer", AutoTokenizer)

    tokenizer = HuggingFaceTokenizer("tokenizer", "immutable", allow_remote=True)

    assert tokenizer.revision == "resolved-commit"
    assert len(tokenizer.fingerprint) == 64
    assert tokenizer.encode("abc", add_special_tokens=True) == [3]
    assert seen["local_files_only"] is False
    assert seen["special"] is True

    with pytest.raises(ValueError, match="immutable"):
        HuggingFaceTokenizer("tokenizer", "main")


def test_compiler_factory_identity_checks_and_input_order(monkeypatch) -> None:
    tokenizer = DeterministicByteTokenizer(tokenizer_id="tokenizer", revision="revision")
    monkeypatch.setattr(compiler_module, "load_tokenizer", lambda *args, **kwargs: tokenizer)

    compiled = ContextPackCompiler.from_pretrained(
        tokenizer_id="tokenizer",
        tokenizer_revision="revision",
        model_id="model",
    )
    pack = compiled.compile(request(order_policy=OrderPolicy.input))

    assert pack.compiled_prefix.index('doc_id="z"') < pack.compiled_prefix.index('doc_id="a"')


@pytest.mark.parametrize(
    ("tokenizer_id", "revision", "fingerprint", "message"),
    [
        ("", "revision", "a" * 64, "tokenizer_id"),
        ("tokenizer", "", "a" * 64, "revision"),
        ("tokenizer", "latest", "a" * 64, "immutable"),
        ("tokenizer", "revision", "short", "fingerprint"),
        ("tokenizer", "revision", "z" * 64, "fingerprint"),
    ],
)
def test_compiler_rejects_unstable_tokenizer_identity(
    tokenizer_id: str,
    revision: str,
    fingerprint: str,
    message: str,
) -> None:
    tokenizer = SimpleNamespace(
        tokenizer_id=tokenizer_id,
        revision=revision,
        fingerprint=fingerprint,
        encode=lambda *args, **kwargs: [1],
    )

    with pytest.raises(ValueError, match=message):
        ContextPackCompiler(tokenizer)


def test_compiler_rejects_pack_from_another_tokenizer() -> None:
    pack = byte_compiler().compile(request())
    other = ContextPackCompiler(
        DeterministicByteTokenizer(tokenizer_id="tokenizer", revision="other"),
        model_id="model",
    )

    with pytest.raises(ValueError, match="identity"):
        other.completion_token_ids(pack, "question")


def test_hashing_rejects_noncanonical_json_and_tokens() -> None:
    with pytest.raises(ValueError, match="empty"):
        canonical_identifier(" ")
    with pytest.raises(ValueError, match="control"):
        canonical_identifier("bad\nid")
    with pytest.raises(ValueError, match="finite"):
        canonical_json_value(math.inf)
    with pytest.raises(TypeError, match="keys"):
        canonical_json_value({1: "value"})
    with pytest.raises(ValueError, match="colliding"):
        canonical_json_value({"Ａ": 1, "A": 2})
    with pytest.raises(TypeError, match="serializable"):
        canonical_json_value({1, 2})
    with pytest.raises(TypeError, match="integers"):
        sha256_token_ids([True])
    with pytest.raises(ValueError, match="unsigned"):
        sha256_token_ids([-1])


def test_configuration_and_completion_edge_validation() -> None:
    assert Settings(vllm_workers="http://one/, http://two").vllm_workers == [
        "http://one",
        "http://two",
    ]
    with pytest.raises(TypeError, match="comma-separated"):
        Settings(vllm_workers=1)
    with pytest.raises(ValidationError, match="whitespace"):
        Settings(cache_salt_secret=f" {'x' * 32}")
    with pytest.raises(ValidationError, match="prompt"):
        CompletionRequest(pack_id="pack", prompt=" \n ")
    with pytest.raises(ValidationError, match="extra_body"):
        CompletionRequest(
            pack_id="pack",
            prompt="question",
            extra_body={f"field-{index}": index for index in range(33)},
        )
