ARG ZEROCLAW_VERSION

FROM debian:trixie-slim AS zeroclaw-release
ARG ZEROCLAW_VERSION

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tar \
    && rm -rf /var/lib/apt/lists/*

RUN test -n "${ZEROCLAW_VERSION}" \
    && arch="$(dpkg --print-architecture)" \
    && case "${arch}" in \
         amd64) zeroclaw_arch="x86_64-unknown-linux-gnu" ;; \
         arm64) zeroclaw_arch="aarch64-unknown-linux-gnu" ;; \
         *) echo "Unsupported architecture: ${arch}" >&2; exit 1 ;; \
       esac \
    && curl -fsSL \
      "https://github.com/zeroclaw-labs/zeroclaw/releases/download/v${ZEROCLAW_VERSION}/zeroclaw-${zeroclaw_arch}.tar.gz" \
      -o /tmp/zeroclaw.tar.gz \
    && mkdir -p /tmp/zeroclaw-extract \
    && tar -xzf /tmp/zeroclaw.tar.gz -C /tmp/zeroclaw-extract \
    && install -m 0755 "$(find /tmp/zeroclaw-extract -type f -name zeroclaw | head -n 1)" /usr/local/bin/zeroclaw \
    && mkdir -p /opt/zeroclaw-runtime \
    && ldd /usr/local/bin/zeroclaw \
      | awk '$2 == "=>" { print $3 } $1 ~ /^\// { print $1 }' \
      | sort -u \
      | xargs -I '{}' sh -c 'mkdir -p "/opt/zeroclaw-runtime$(dirname "{}")" && cp "{}" "/opt/zeroclaw-runtime{}"' \
    && tar -C /opt/zeroclaw-runtime -cf /tmp/zeroclaw-runtime.tar .

FROM agent-harness-eval-base:latest

COPY --from=zeroclaw-release /tmp/zeroclaw-runtime.tar /tmp/zeroclaw-runtime.tar
RUN tar -C / -xf /tmp/zeroclaw-runtime.tar && rm /tmp/zeroclaw-runtime.tar
COPY --from=zeroclaw-release /usr/local/bin/zeroclaw /usr/local/bin/zeroclaw
