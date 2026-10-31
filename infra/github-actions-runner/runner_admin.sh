#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

repository=${RUNNER_REPOSITORY:-JohnWick2ETH/llm-learning-journey}
container_name=cs336-gpu-runner
export RUNNER_URL="https://github.com/${repository}"
runner_version=2.338.2
runner_archive="actions-runner-linux-x64-${runner_version}.tar.gz"
runner_sha256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40

if [[ -n "${RUNNER_PROXY:-}" ]]; then
  export RUNNER_NETWORK_MODE=host
fi

usage() {
  cat <<'EOF'
Usage: ./runner_admin.sh COMMAND [OPTIONS]

Commands:
  start                 Build, register if needed, and start the runner
  status                Show whether the runner is stopped, idle, or busy
  logs [COMPOSE OPTIONS]
                        Show the last 100 log lines; accepts --follow/--tail
  deactivate            Stop and deregister the runner from GitHub
  help                  Show this help

Environment:
  RUNNER_REPOSITORY     GitHub OWNER/REPO (default: JohnWick2ETH/llm-learning-journey)
  RUNNER_NAME           Runner name (default: cs336-gpu-runner)
  RUNNER_PROXY          Runtime HTTP proxy, for example http://127.0.0.1:1080
  RUNNER_TOKEN          Optional short-lived registration token
  RUNNER_REMOVE_TOKEN   Optional short-lived removal token
EOF
}

container_exists() {
  docker inspect "${container_name}" >/dev/null 2>&1
}

container_is_running() {
  [[ "$(docker inspect --format '{{.State.Running}}' "${container_name}" 2>/dev/null || true)" == "true" ]]
}

runner_is_busy() {
  container_is_running \
    && docker top "${container_name}" -eo comm 2>/dev/null \
      | grep -q '^Runner.Worker$'
}

request_token() {
  local token_kind=$1
  local api_path=$2
  local token=""

  echo "Requesting a short-lived runner ${token_kind} token..." >&2
  if token=$(gh api --method POST "${api_path}" --jq .token); then
    printf '%s' "${token}"
    return 0
  fi

  if [[ ! -t 0 ]]; then
    echo "Runner ${token_kind} requires repository administration permission." >&2
    return 1
  fi

  read -r -s -p "Paste a short-lived runner ${token_kind} token: " token
  echo >&2
  printf '%s' "${token}"
}

show_status() {
  if ! container_exists; then
    echo "Runner container: not created"
    echo "GitHub registration: unknown"
    return 0
  fi

  local state
  state=$(docker inspect --format '{{.State.Status}}' "${container_name}")
  echo "Runner container: ${state}"

  if [[ "${state}" != "running" ]]; then
    echo "Runner activity: stopped"
    return 0
  fi

  if runner_is_busy; then
    echo "Runner activity: busy"
  else
    echo "Runner activity: idle"
  fi

  if docker exec "${container_name}" test -f /home/runner/actions-runner/.runner; then
    echo "Local registration: configured"
  else
    echo "Local registration: not configured"
  fi

  echo "Latest log:"
  docker logs --tail 1 "${container_name}" 2>&1 | sed 's/^/  /'
}

start_runner() {
  if container_is_running; then
    echo "Runner is already running; leaving the active container unchanged."
    show_status
    return 0
  fi

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

  if docker compose run --rm --no-deps --entrypoint test runner \
      -f /home/runner/actions-runner/.runner; then
    echo "Using the existing runner registration."
    RUNNER_TOKEN=""
  elif [[ -n "${RUNNER_TOKEN:-}" ]]; then
    echo "Using the supplied short-lived runner registration token."
  else
    RUNNER_TOKEN=$(request_token registration \
      "repos/${repository}/actions/runners/registration-token")
  fi
  export RUNNER_TOKEN

  docker compose up --detach --no-build
  unset RUNNER_TOKEN

  echo "Waiting for the runner to become ready..."
  for _ in $(seq 1 30); do
    if docker compose logs --no-color runner 2>&1 | grep -q "Listening for Jobs"; then
      show_status
      return 0
    fi
    sleep 2
  done

  docker compose logs --no-color runner
  echo "Runner did not become ready within 60 seconds." >&2
  return 1
}

show_logs() {
  if ! container_exists; then
    echo "Runner container does not exist." >&2
    return 1
  fi

  if [[ $# -eq 0 ]]; then
    docker compose logs --no-color --tail 100 runner
  else
    docker compose logs --no-color "$@" runner
  fi
}

deactivate_runner() {
  if runner_is_busy; then
    echo "Runner is processing a job; refusing to deactivate it." >&2
    echo "Wait for the job to finish or cancel it on GitHub first." >&2
    return 1
  fi

  local remove_token=${RUNNER_REMOVE_TOKEN:-}
  if [[ -z "${remove_token}" ]]; then
    remove_token=$(request_token removal \
      "repos/${repository}/actions/runners/remove-token")
  else
    echo "Using the supplied short-lived runner removal token."
  fi

  local was_running=false
  if container_is_running; then
    was_running=true
    docker compose stop runner
  fi

  if ! docker compose run --rm --no-deps --entrypoint gosu runner \
      runner ./config.sh remove --token "${remove_token}"; then
    echo "Runner deregistration failed." >&2
    if [[ "${was_running}" == "true" ]]; then
      echo "Restarting the existing runner container." >&2
      docker compose start runner
    fi
    return 1
  fi

  remove_token=""
  unset RUNNER_REMOVE_TOKEN || true
  docker compose rm --force --stop runner
  echo "Runner deactivated and removed from GitHub."
  echo "The runner image and named volumes were preserved."
}

command=${1:-help}
if [[ $# -gt 0 ]]; then
  shift
fi

case "${command}" in
  start)
    [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    start_runner
    ;;
  status)
    [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    show_status
    ;;
  logs)
    show_logs "$@"
    ;;
  deactivate)
    [[ $# -eq 0 ]] || { usage >&2; exit 2; }
    deactivate_runner
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
