FROM ghcr.io/astral-sh/uv:0.11.6-python3.13-trixie@sha256:b3c543b6c4f23a5f2df22866bd7857e5d304b67a564f4feab6ac22044dde719b AS uv_source
FROM tianon/gosu:1.19-trixie@sha256:3b176695959c71e123eb390d427efc665eeb561b1540e82679c15e992006b8b9 AS gosu_source
FROM debian:13.4

# Disable Python stdout buffering to ensure logs are printed immediately
ENV PYTHONUNBUFFERED=1

# Store Playwright browsers outside /opt/hermes so the build-time
# install survives both the /opt/data volume overlay and the source
# bind-mount at /opt/hermes used in dev environments.
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

# Install system dependencies in one layer, clear APT cache
# tini reaps orphaned zombie processes (MCP stdio subprocesses, git, bun, etc.)
# that would otherwise accumulate when hermes runs as PID 1. See #15012.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential nodejs npm python3 ripgrep ffmpeg gcc python3-dev libffi-dev procps git \
        openssh-client docker-cli tini \
        curl gpg vim-tiny less jq file sqlite3 tree imagemagick && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
         -o /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
         > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for runtime; UID can be overridden via HERMES_UID at runtime
RUN useradd -u 10000 -m -d /opt/data hermes

COPY --chmod=0755 --from=gosu_source /gosu /usr/local/bin/
COPY --chmod=0755 --from=uv_source /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/

# google-workspace スキルの OAuth 認証で動的 pip install が
# Debian externally-managed-environment に弾かれるのを回避
RUN uv pip install \
      --python /usr/bin/python3 \
      --target /opt/google-libs \
      --no-cache \
      google-api-python-client google-auth-oauthlib google-auth-httplib2
ENV PYTHONPATH=/opt/google-libs

WORKDIR /opt/hermes

# ---------- Layer-cached dependency install ----------
# Copy only package manifests first so npm install + Playwright are cached
# unless the lockfiles themselves change.
COPY package.json package-lock.json ./
COPY web/package.json web/package-lock.json web/
COPY ui-tui/package.json ui-tui/package-lock.json ui-tui/
COPY ui-tui/packages/hermes-ink/package.json ui-tui/packages/hermes-ink/package-lock.json ui-tui/packages/hermes-ink/

RUN npm install --prefer-offline --no-audit && \
    npx playwright install --with-deps chromium --only-shell && \
    (cd web && npm install --prefer-offline --no-audit) && \
    (cd ui-tui && npm install --prefer-offline --no-audit) && \
    npm cache clean --force

# ---------- Source code ----------
# .dockerignore excludes node_modules, so the installs above survive.
COPY --chown=hermes:hermes . .

# Build browser dashboard and terminal UI assets.
RUN cd web && npm run build && \
    cd ../ui-tui && npm run build && \
    rm -rf node_modules/@hermes/ink && \
    rm -rf packages/hermes-ink/node_modules && \
    cp -R packages/hermes-ink node_modules/@hermes/ink && \
    npm install --omit=dev --prefer-offline --no-audit --prefix node_modules/@hermes/ink && \
    rm -rf node_modules/@hermes/ink/node_modules/react && \
    node --input-type=module -e "await import('@hermes/ink')"

# ---------- Permissions ----------
# Make install dir world-readable so any HERMES_UID can read it at runtime.
# The venv needs to be traversable too.
USER root
RUN chmod -R a+rX /opt/hermes
# Start as root so the entrypoint can usermod/groupmod + gosu.
# If HERMES_UID is unset, the entrypoint drops to the default hermes user (10000).

# ---------- Python virtualenv (/opt/venv 外出し、source mount 対応) ----------
RUN mkdir -p /opt/venv && chown hermes:hermes /opt/venv
RUN chown -R hermes:hermes /opt/hermes
USER hermes
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:$PATH" \
    uv pip install --no-cache-dir -e ".[all]"

USER root
RUN chmod +x /opt/hermes/docker/entrypoint.sh

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV HERMES_WEB_DIST=/opt/hermes/hermes_cli/web_dist
ENV HERMES_HOME=/opt/data
ENV PATH="/opt/data/.local/bin:${PATH}"
VOLUME [ "/opt/data" ]
ENTRYPOINT [ "/usr/bin/tini", "-g", "--", "/opt/hermes/docker/entrypoint.sh" ]
