# voice-agent

Self-hosted real-time healthcare voice agent: STT (Nemotron) → LLM (Qwen3.8-27B on vLLM) → TTS (Fish S2 Pro),
all on one GPU server.

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

Then point the `verda` alias in `~/.ssh/config` at the new IP and deploy.

## Deploy (every change)

Commit and push locally, then:

```
ssh verda "tmux new -d -s deploy 'cd /srv/voice && ./scripts/deploy.sh > /tmp/deploy.log 2>&1'"
ssh verda "tail -n 20 /tmp/deploy.log"
```

`deploy.sh` pulls, downloads missing weights, runs `docker compose up -d`, waits for every service to be
healthy, and runs `scripts/smoke_test.sh`. The last line is `Deploy OK: <commit>`, or it prints the failing
service's logs.
