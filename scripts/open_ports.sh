#!/usr/bin/env bash
# Opens only the public ports of one GPU cell in ufw; everything else stays closed. Run as root
# (bootstrap.sh calls it). Safe to re-run. Containers on host networking are covered by these rules;
# published Docker ports would bypass ufw, which is why every other port binds to 127.0.0.1.
set -euo pipefail

ufw allow OpenSSH comment 'admin SSH, keys only'
ufw allow 80/tcp comment 'Caddy: Lets Encrypt HTTP challenge, redirect to HTTPS'
ufw allow 443/tcp comment 'Caddy: HTTPS and secure WebSocket to LiveKit signaling'
ufw allow 7881/tcp comment 'LiveKit: WebRTC over TCP when a network blocks UDP'
ufw allow 50000:60000/udp comment 'LiveKit: WebRTC media over UDP'
ufw --force enable
ufw status verbose
