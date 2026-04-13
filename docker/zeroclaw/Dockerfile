FROM agent-harness-eval-base:latest

# ZeroClaw requires GLIBC >= 2.39 (Debian Trixie).
# Copy the binary and its required libs from a Trixie stage.
COPY --from=debian:trixie-slim /lib/x86_64-linux-gnu/libc.so.6 /lib/x86_64-linux-gnu/libc.so.6
COPY --from=debian:trixie-slim /lib/x86_64-linux-gnu/libm.so.6 /lib/x86_64-linux-gnu/libm.so.6
COPY --from=debian:trixie-slim /lib/x86_64-linux-gnu/libpthread.so.0 /lib/x86_64-linux-gnu/libpthread.so.0
COPY --from=debian:trixie-slim /lib64/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2

COPY zeroclaw /usr/local/bin/zeroclaw
RUN chmod +x /usr/local/bin/zeroclaw
