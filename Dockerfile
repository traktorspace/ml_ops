FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install git and ssh
RUN apt-get update && \
    apt-get install -y --no-install-recommends git openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Trust Git server
RUN mkdir -p /root/.ssh && \
ssh-keyscan github.com >> /root/.ssh/known_hosts && \
chmod 644 /root/.ssh/known_hosts

RUN --mount=type=ssh ssh -T git@github.com || true

# Install dependencies with uv sync using SSH agent
RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /app

RUN --mount=type=ssh \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

RUN pre-commit install

ENTRYPOINT []
