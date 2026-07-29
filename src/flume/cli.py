from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.table import Table

from flume.config import Settings
from flume.models import CompletionRequest, PackRegistrationRequest
from flume.sdk import FlumeClient, chunks_from_files

app = typer.Typer(help="Flume RAG cache compiler and vLLM proxy.")
pack_app = typer.Typer(help="Create and inspect context packs.")
app.add_typer(pack_app, name="pack")
console = Console()


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8080,
    database_url: Annotated[
        str,
        typer.Option(help="SQLAlchemy database URL."),
    ] = "sqlite:///./flume.db",
    vllm_worker: Annotated[
        list[str] | None,
        typer.Option(help="vLLM OpenAI-compatible worker URL. Repeat for multiple workers."),
    ] = None,
    model: Annotated[str, typer.Option(help="Default model id.")] = "local-model",
    tokenizer: Annotated[str, typer.Option(help="Default tokenizer id.")] = "local-tokenizer",
    allow_remote_tokenizer: Annotated[
        bool,
        typer.Option(help="Allow Hugging Face tokenizer downloads."),
    ] = False,
) -> None:
    from flume.api import create_app

    settings = Settings(
        host=host,
        port=port,
        database_url=database_url,
        vllm_workers=vllm_worker or ["http://localhost:8000"],
        model_id=model,
        tokenizer_id=tokenizer,
        allow_remote_tokenizer=allow_remote_tokenizer,
    )
    uvicorn.run(create_app(settings), host=host, port=port)


@pack_app.command("create")
def create_pack(
    files: Annotated[
        list[Path],
        typer.Argument(help="Text files to compile into the context pack."),
    ],
    tenant: Annotated[str, typer.Option(help="Tenant id.")] = "default",
    template_id: Annotated[str, typer.Option(help="Template id.")] = "default-rag-v1",
    api: Annotated[str, typer.Option(help="Flume API URL.")] = "http://localhost:8080",
) -> None:
    chunks = chunks_from_files(files)
    request = PackRegistrationRequest(
        template_id=template_id,
        chunks=chunks,
    )
    with FlumeClient(api, tenant_id=tenant) as client:
        pack = client.register_pack(request)
    console.print(pack.model_dump_json(indent=2))


@pack_app.command("list")
def list_packs(
    tenant: Annotated[str, typer.Option(help="Tenant id.")] = "default",
    api: Annotated[str, typer.Option(help="Flume API URL.")] = "http://localhost:8080",
) -> None:
    with FlumeClient(api, tenant_id=tenant) as client:
        page = client.list_packs()
    table = Table(title="Context Packs")
    table.add_column("pack_id")
    table.add_column("model")
    table.add_column("tokens", justify="right")
    table.add_column("created_at")
    for pack in page.items:
        table.add_row(
            pack.pack_id,
            pack.model_id,
            str(pack.token_count),
            str(pack.created_at),
        )
    console.print(table)


@app.command()
def warm(
    pack_id: str,
    tenant: Annotated[str, typer.Option(help="Tenant id.")] = "default",
    api: Annotated[str, typer.Option(help="Flume API URL.")] = "http://localhost:8080",
) -> None:
    with FlumeClient(api, tenant_id=tenant) as client:
        response = client.warm_pack(pack_id)
    console.print(response.model_dump_json(indent=2))


@app.command()
def ask(
    pack_id: str,
    prompt: str,
    tenant: Annotated[str, typer.Option(help="Tenant id.")] = "default",
    api: Annotated[str, typer.Option(help="Flume API URL.")] = "http://localhost:8080",
    max_tokens: Annotated[int, typer.Option(help="Generation max tokens.")] = 256,
) -> None:
    with FlumeClient(api, tenant_id=tenant) as client:
        response = client.complete(
            CompletionRequest(pack_id=pack_id, prompt=prompt, max_tokens=max_tokens)
        )
    console.print(response.choices[0].text if response.choices else "")
    console.print(response.model_dump_json(indent=2))


@app.command()
def stats(
    tenant: Annotated[str, typer.Option(help="Tenant id.")] = "default",
    api: Annotated[str, typer.Option(help="Flume API URL.")] = "http://localhost:8080",
) -> None:
    with FlumeClient(api, tenant_id=tenant) as client:
        response = client.stats()
    console.print(response.model_dump_json(indent=2))


def main() -> None:
    app()
