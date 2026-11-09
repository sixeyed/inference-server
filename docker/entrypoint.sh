#!/bin/sh
# Put the SSH material where OpenSSH will accept it, then start the engine.
#
# Secret and bind mounts arrive root-owned or group-readable, and ssh refuses a
# private key anyone else can read - so the mounted files are copied into
# ~/.ssh with 0600 rather than used where they are mounted.
set -eu

src=/etc/inference/ssh
mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"
for name in id_ed25519 known_hosts; do
    if [ -f "$src/$name" ]; then
        cp "$src/$name" "$HOME/.ssh/$name"
        chmod 600 "$HOME/.ssh/$name"
    fi
done

exec python -m inference engine \
    --usage-db /data/usage.db \
    --trace-dir /data/traces \
    --log-dir /data/logs \
    --settings /data/settings.json \
    --trust-proxy \
    "$@"
