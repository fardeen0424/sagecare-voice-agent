# voice-agent

Self-hosted real-time healthcare voice agent. All three models run on one GPU server
([infra/docker-compose.yml](infra/docker-compose.yml)), each bound to `127.0.0.1`:

| Service | Model | Server | Port |
|---|---|---|---|
| `llm` | Qwen3.8-27B FP8 | vLLM 0.31.0 (OpenAI API) | 8000 |
| `stt` | Nemotron 3.5 ASR streaming | our WebSocket wrapper ([infra/stt](infra/stt)) | 8001 |
| `tts` | Fish S2 Pro | SGLang-Omni 0.1.8 (`/v1/audio/speech`) | 8002 |

## New GPU server (Ubuntu 24.04, NVIDIA driver >= 580)

From the laptop, logged in as the provider's root user:

```
scp scripts/bootstrap.sh root@NEW_IP:/root/
ssh -t root@NEW_IP "bash /root/bootstrap.sh"
```

It installs automatic security updates, fail2ban, Docker and the NVIDIA Container Toolkit. It creates the
`fardeen` user and clones this repo with a read-only deploy key. It writes `.env` (it asks for `HF_TOKEN`),
allows only SSH through ufw, and turns off root and password login. Re-run it after any interruption: it
skips steps that are already done. If it installs a driver, reboot and run it again.

It ends by starting `deploy.sh` in the background, so the three models come up on their own. Point the
`verda` alias in `~/.ssh/config` at the new IP and follow `ssh verda "tail -n 20 /tmp/deploy.log"`.

## Deploy (every change)

Commit and push locally, then:

```
ssh verda "tmux new -d -s deploy 'cd /srv/voice && ./scripts/deploy.sh > /tmp/deploy.log 2>&1'"
ssh verda "tail -n 20 /tmp/deploy.log"
```

`deploy.sh` pulls, downloads missing weights, builds and starts the services, waits until all are healthy,
and runs one round of `scripts/check.py`. The last line is `Deploy OK: <commit>`, or it prints the failing
service's logs.

## Check and benchmark

`scripts/check.py` fails unless the LLM replies and calls a tool and STT hears back what TTS said, then
prints p50/p95 latency for each model over N rounds:

```
ssh verda "cd /srv/voice && docker compose -f infra/docker-compose.yml exec -T stt python - 20 < scripts/check.py"
```
