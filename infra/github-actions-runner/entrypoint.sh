#!/usr/bin/env bash
set -euo pipefail

runner_home=/home/runner/actions-runner
runner_template=/opt/actions-runner

if [[ ! -x "${runner_home}/run.sh" ]]; then
  cp -a "${runner_template}/." "${runner_home}/"
fi

chown -R runner:runner "${runner_home}" /home/runner/_work

if [[ ! -f "${runner_home}/.runner" ]]; then
  : "${RUNNER_URL:?Set RUNNER_URL to the repository URL}"
  : "${RUNNER_TOKEN:?Set RUNNER_TOKEN to a short-lived registration token}"

  gosu runner ./config.sh \
    --unattended \
    --url "${RUNNER_URL}" \
    --token "${RUNNER_TOKEN}" \
    --name "${RUNNER_NAME:-cs336-gpu-runner}" \
    --labels "${RUNNER_LABELS:-gpu,triton}" \
    --work /home/runner/_work \
    --replace \
    --disableupdate
fi

unset RUNNER_TOKEN
exec gosu runner ./run.sh
