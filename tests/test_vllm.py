import httpx
import pytest

from flume.vllm import VLLMClient, VLLMUpstreamError


class FragmentedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_completion_sends_integer_tokens_and_rejects_reserved_fields() -> None:
    payloads = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"text": "answer", "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    client = VLLMClient(transport=httpx.MockTransport(handler))
    await client.start()
    result = await client.complete(
        worker_url="http://worker",
        model="model",
        prompt=[1, 2, 3],
        max_tokens=1,
        temperature=0,
        cache_salt="salt",
    )

    assert result.text == "answer"
    assert b'"prompt":[1,2,3]' in payloads[0]
    with pytest.raises(ValueError, match="reserved"):
        await client.complete(
            worker_url="http://worker",
            model="model",
            prompt=[1],
            max_tokens=1,
            temperature=0,
            extra_body={"prompt": "override"},
        )
    await client.close()


@pytest.mark.asyncio
async def test_stream_is_prevalidated_and_forwards_raw_bytes() -> None:
    raw = b'data: {"choices":[{"text":"hi"}]}\n\ndata: [DONE]\n\n'

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=FragmentedStream([raw[:9], raw[9:31], raw[31:]]),
            headers={"content-type": "text/event-stream"},
        )

    first_tokens = []
    completions = []
    client = VLLMClient(transport=httpx.MockTransport(handler))
    await client.start()
    stream = await client.open_stream_completion(
        worker_url="http://worker",
        model="model",
        prompt=[1, 2],
        max_tokens=1,
        temperature=0,
        on_first_token=first_tokens.append,
        on_done=completions.append,
    )

    chunks = [chunk async for chunk in stream]

    assert b"".join(chunks) == raw
    assert len(first_tokens) == 1
    assert len(completions) == 1
    await client.close()


@pytest.mark.asyncio
async def test_stream_error_is_known_before_downstream_headers() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"sensitive upstream body")

    client = VLLMClient(transport=httpx.MockTransport(handler))
    await client.start()

    with pytest.raises(VLLMUpstreamError, match="HTTP 503") as error:
        await client.open_stream_completion(
            worker_url="http://worker",
            model="model",
            prompt=[1],
            max_tokens=1,
            temperature=0,
        )

    assert "sensitive" not in str(error.value)
    await client.close()
