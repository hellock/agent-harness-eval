ARG ZEROCLAW_VERSION

FROM debian:trixie-slim AS zeroclaw-release
ARG ZEROCLAW_VERSION

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "${ZEROCLAW_VERSION}" \
    && curl -fsSL \
      "https://github.com/zeroclaw-labs/zeroclaw/releases/download/v${ZEROCLAW_VERSION}/zeroclaw-x86_64-unknown-linux-gnu.tar.gz" \
      -o /tmp/zeroclaw.tar.gz \
    && mkdir -p /tmp/zeroclaw-extract \
    && tar -xzf /tmp/zeroclaw.tar.gz -C /tmp/zeroclaw-extract \
    && install -m 0755 "$(find /tmp/zeroclaw-extract -type f -name zeroclaw | head -n 1)" /usr/local/bin/zeroclaw

FROM agent-harness-eval-base:latest

# ZeroClaw requires GLIBC >= 2.39 (Debian Trixie).
# Copy the binary and its required libs from a Trixie stage.
COPY --from=debian:trixie-slim /lib/x86_64-linux-gnu/libc.so.6 /lib/x86_64-linux-gnu/libc.so.6
COPY --from=debian:trixie-slim /lib/x86_64-linux-gnu/libm.so.6 /lib/x86_64-linux-gnu/libm.so.6
COPY --from=debian:trixie-slim /lib/x86_64-linux-gnu/libpthread.so.0 /lib/x86_64-linux-gnu/libpthread.so.0
COPY --from=debian:trixie-slim /lib64/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2

COPY --from=zeroclaw-release /usr/local/bin/zeroclaw /usr/local/bin/zeroclaw
