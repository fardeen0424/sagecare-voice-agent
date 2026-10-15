#!/usr/bin/env bash
# One-time setup of a brand-new Ubuntu 24.04 GPU server. Safe to re-run: every step skips work already done.
#
# From the laptop, as root on the new server (the provider's initial login):
#   scp scripts/bootstrap.sh root@NEW_IP:/root/
#   ssh -t root@NEW_IP "bash /root/bootstrap.sh"
# It asks for a sudo password for the user, for adding the deploy key on GitHub, and for HF_TOKEN
# (or pass it: ssh -t root@NEW_IP "HF_TOKEN=... bash /root/bootstrap.sh"). It ends by starting the deploy.
# At the end root login and password login are disabled; reconnect as the user.
set -euo pipefail

USER_NAME=${USER_NAME:-fardeen}
REPO=${REPO:-git@github.com:fardeen0424/sagecare-voice-agent.git}
APP=/srv/voice
MIN_DRIVER=580   # the vLLM and SGLang images are built on CUDA 13.0, which needs driver >= 580

say () { echo -e "\n==> $*"; }
as_user () { sudo -u "$USER_NAME" -H bash -c "$1"; }
[ "$EUID" -eq 0 ] || { echo "Run as root (or with sudo)."; exit 1; }
export DEBIAN_FRONTEND=noninteractive

say "Base packages + automatic security updates"
apt-get update -q
apt-get install -yq git curl ca-certificates gnupg tmux ufw fail2ban unattended-upgrades python3 ubuntu-drivers-common
printf 'APT::Periodic::Update-Package-Lists "1";\nAPT::Periodic::Unattended-Upgrade "1";\n' \
  > /etc/apt/apt.conf.d/20auto-upgrades
systemctl enable --now fail2ban   # Ubuntu's package enables the sshd jail by default

say "NVIDIA driver (need >= $MIN_DRIVER)"
if ! nvidia-smi >/dev/null 2>&1; then
  ubuntu-drivers install --gpgpu   # picks the right (open) driver, also for Blackwell cards
  echo "Driver installed. Reboot, then run this script again."; exit 0
fi
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
[ "${driver%%.*}" -ge "$MIN_DRIVER" ] || { echo "Driver $driver is too old, need >= $MIN_DRIVER."; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

say "Docker"
if ! command -v docker >/dev/null; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]" \
    "https://download.docker.com/linux/ubuntu $codename stable" > /etc/apt/sources.list.d/docker.list
  apt-get update -q
  apt-get install -yq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

say "NVIDIA Container Toolkit"
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q
  apt-get install -yq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
fi
docker run --rm --gpus all \
  nvidia/cuda:12.8.0-base-ubuntu24.04@sha256:e778509d37d66475120929671500377524f7278478ba08131b07ef3ffcc0dce0 nvidia-smi -L

say "User $USER_NAME (key-only SSH, sudo with password, docker group)"
id "$USER_NAME" >/dev/null 2>&1 || adduser --disabled-password --gecos "" "$USER_NAME"
usermod -aG sudo,docker "$USER_NAME"
home=$(getent passwd "$USER_NAME" | cut -d: -f6)
if [ ! -s "$home/.ssh/authorized_keys" ]; then   # reuse the laptop key root was given
  install -d -m 700 -o "$USER_NAME" -g "$USER_NAME" "$home/.ssh"
  install -m 600 -o "$USER_NAME" -g "$USER_NAME" /root/.ssh/authorized_keys "$home/.ssh/authorized_keys"
fi
if [ "$(passwd -S "$USER_NAME" | awk '{print $2}')" != "P" ]; then
  [ -t 0 ] || { echo "No sudo password set. Re-run with: ssh -t ..."; exit 1; }
  echo "Choose a sudo password for $USER_NAME:"; passwd "$USER_NAME"
fi

say "Folders: /models (weights) and $APP (repo)"
install -d -o "$USER_NAME" -g "$USER_NAME" /models "$APP"

say "Read-only GitHub deploy key"
if [ ! -f "$home/.ssh/github_deploy" ]; then
  as_user "ssh-keygen -q -t ed25519 -N '' -C 'voice-agent deploy key' -f ~/.ssh/github_deploy"
  as_user "cat >> ~/.ssh/config && chmod 600 ~/.ssh/config" <<'EOF'
Host github.com
    IdentityFile ~/.ssh/github_deploy
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
EOF
fi
until as_user "git ls-remote $REPO >/dev/null 2>&1"; do
  echo "Add this key on GitHub: repo > Settings > Deploy keys > Add (leave write access off):"
  cat "$home/.ssh/github_deploy.pub"
  [ -t 0 ] || exit 1
  read -rp "Press Enter once it is added... "
done
[ -d "$APP/.git" ] || as_user "git clone $REPO $APP"

say "Secrets ($APP/.env)"
if [ ! -f "$APP/.env" ]; then
  install -m 600 -o "$USER_NAME" -g "$USER_NAME" "$APP/.env.example" "$APP/.env"
  token=${HF_TOKEN:-}
  if [ -z "$token" ] && [ -t 0 ]; then
    read -rsp "HF_TOKEN (Hugging Face read token): " token; echo
  fi
  sed -i "s|^HF_TOKEN=.*|HF_TOKEN=$token|" "$APP/.env"
fi

say "Firewall: SSH only (later phases open 443, SIP, RTP explicitly)"
ufw allow OpenSSH
ufw --force enable

say "SSH hardening: no root login, keys only"
cat > /etc/ssh/sshd_config.d/01-voice-hardening.conf <<'EOF'
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
EOF
sshd -t && systemctl reload ssh

say "Deploying in the background (downloads weights, builds, starts, checks; ~20 min on a new server)"
as_user "tmux new -d -s deploy 'cd $APP && ./scripts/deploy.sh > /tmp/deploy.log 2>&1'"
echo "Server ready. Reconnect as $USER_NAME and follow: tail -n 20 /tmp/deploy.log (ends with 'Deploy OK')."
