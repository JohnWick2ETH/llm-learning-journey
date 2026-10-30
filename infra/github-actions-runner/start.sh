#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

repository=${RUNNER_REPOSITORY:-JohnWick2ETH/llm-learning-journey}
export RUNNER_URL="https://github.com/${repository}"
runner_version=2.338.2
runner_archive="actions-runner-linux-x64-${runner_version}.tar.gz"
runner_sha256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40

echo "Checking NVIDIA GPU passthrough..."
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi >/dev/null

mkdir -p .cache
if [[ ! -f ".cache/${runner_archive}" ]]; then
  echo "Downloading GitHub Actions runner ${runner_version}..."
  gh release download "v${runner_version}" \
    --repo actions/runner \
    --pattern "${runner_archive}" \
    --dir .cache
fi
echo "${runner_sha256}  .cache/${runner_archive}" | sha256sum --check --strict

docker compose build runner

if [[ -z "${RUNNER_TOKEN:-}" ]]; then
  echo "Requesting a short-lived runner registration token..."
  if ! RUNNER_TOKEN=$(gh api \
      --method POST \
      "repos/${repository}/actions/runners/registration-token" \
      --jq .token); then
    if [[ ! -t 0 ]]; then
      echo "Runner registration requires repository administration permission." >&2
      exit 1
    fi
    read -r -s -p "Paste a short-lived runner registration token: " RUNNER_TOKEN
    echo
  fi
else
  echo "Using the supplied short-lived runner registration token."
fi
export RUNNER_TOKEN

docker compose up --detach --no-build
unset RUNNER_TOKEN

echo "Waiting for the runner to become ready..."
for _ in $(seq 1 30); do
  if docker compose logs --no-color runner 2>&1 | grep -q "Listening for Jobs"; then
    docker compose ps
    exit 0
  fi
  sleep 2
done

docker compose logs --no-color runner
echo "Runner did not become ready within 60 seconds." >&2
exit 1
