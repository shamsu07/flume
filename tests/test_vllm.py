import httpx
import pytest

from flume.vllm import VLLMClient, VLLMConnectionError, VLLMUpstreamError


class FragmentedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(200, True), (204, True), (302, False), (401, False), (503, False)],
)
async def test_health_requires_a_2xx_response(status_code: int, expected: bool) -> None:
    client = VLLMClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status_code)),
    )
    await client.start()

    assert await client.health("http://worker") is expected
    await client.close()


@pytest.mark.asyncio
async def test_load_parses_and_sums_vllm_running_and_waiting_metrics() -> None:
    metrics = """
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="a"} 2
vllm:num_requests_running{model_name="b"} 1
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="a"} 4
vllm:num_requests_waiting{model_name="b"} 2
"""
    client = VLLMClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=metrics)),
    )
    await client.start()

    load = await client.load("http://worker")

    assert load is not None
    assert load.running == 3
    assert load.waiting == 6
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metrics",
    [
        "vllm:num_requests_running 1\n",
        "vllm:num_requests_running -1\nvllm:num_requests_waiting 0\n",
        "not prometheus",
    ],
)
async def test_load_rejects_missing_or_invalid_metrics(metrics: str) -> None:
    client = VLLMClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text=metrics)),
    )
    await client.start()

    assert await client.load("http://worker") is None
    await client.close()


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


@pytest.mark.asyncio
async def test_stream_inspection_handles_comments_crlf_and_fragmentation() -> None:
    raw = (
        b": keepalive\r\n\r\n"
        b"event: ping\r\n\r\n"
        b'data: {"choices":[{"text":"token"}]}\r\n\r\n'
        b"data: [DONE]\r\n\r\n"
    )

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=FragmentedStream([raw[:3], raw[3:19], raw[19:47], raw[47:71], raw[71:]]),
            headers={"content-type": "text/event-stream"},
        )

    first_tokens: list[float] = []
    client = VLLMClient(transport=httpx.MockTransport(handler))
    await client.start()
    stream = await client.open_stream_completion(
        worker_url="http://worker",
        model="model",
        prompt=[1],
        max_tokens=1,
        temperature=0,
        on_first_token=first_tokens.append,
    )

    forwarded = b"".join([chunk async for chunk in stream])

    assert forwarded == raw
    assert len(first_tokens) == 1
    assert not VLLMClient.line_has_token(": keepalive")
    assert not VLLMClient.line_has_token("event: ping")
    assert not VLLMClient.line_has_token("data: [DONE]")
    await client.close()


@pytest.mark.asyncio
async def test_closing_stream_closes_upstream_and_records_done_once() -> None:
    class ClosingStream(httpx.AsyncByteStream):
        def __init__(self) -> None:
            self.closed = False

        async def __aiter__(self):
            yield b": keepalive\n\n"

        async def aclose(self) -> None:
            self.closed = True

    upstream = ClosingStream()
    completions: list[float] = []

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=upstream)

    client = VLLMClient(transport=httpx.MockTransport(handler))
    await client.start()
    stream = await client.open_stream_completion(
        worker_url="http://worker",
        model="model",
        prompt=[1],
        max_tokens=1,
        temperature=0,
        on_done=completions.append,
    )

    await stream.aclose()
    await stream.aclose()

    assert upstream.closed
    assert len(completions) == 1
    await client.close()


@pytest.mark.asyncio
async def test_completion_leaves_the_single_retry_to_api_failover() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("unavailable", request=request)

    client = VLLMClient(transport=httpx.MockTransport(handler))
    await client.start()

    with pytest.raises(VLLMConnectionError):
        await client.complete(
            worker_url="http://worker",
            model="model",
            prompt=[1],
            max_tokens=1,
            temperature=0,
        )

    assert calls == 1
    await client.close()
