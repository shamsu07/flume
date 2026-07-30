FROM python:3.12-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --gid 10001 flume \
    && useradd --uid 10001 --gid flume --no-create-home --shell /usr/sbin/nologin flume

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src ./src

RUN pip install --no-cache-dir . \
    && mkdir -p /data \
    && chown flume:flume /data

USER 10001:10001

EXPOSE 8080

STOPSIGNAL SIGTERM

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/livez', timeout=2)"]

CMD ["flume", "serve", "--host", "0.0.0.0", "--port", "8080"]
