# Local GPU GitHub Actions runner

This directory runs a repository-scoped GitHub Actions runner in Docker with
the host NVIDIA GPU passed through. The runner has the custom labels `gpu` and
`triton`; GitHub also gives it the standard `self-hosted`, `linux`, and `x64`
labels.

## Prerequisites

- An NVIDIA driver and NVIDIA Container Toolkit on the host.
- Docker Engine with Compose 2.30 or newer.
- GitHub CLI (`gh`) authenticated as a repository administrator.

The container deliberately does not mount `/var/run/docker.sock`. Jobs can use
the GPU, network, and their workspace, but cannot control the host Docker daemon.

## Start

From this directory, run:

```bash
./start.sh
```

`start.sh` checks GPU passthrough, downloads and checksum-verifies the pinned
official Actions runner, builds its image, requests a one-hour registration
token through the authenticated `gh` CLI, and waits until GitHub reports that it
is listening for jobs. Runner state and job workspaces live in named Docker
volumes, so a normal container restart does not require registration again.

Creating a registration token requires repository administration permission. If
the authenticated `gh` account lacks that permission, `start.sh` prompts for a
short-lived token. A repository administrator can obtain one under **Settings →
Actions → Runners → New self-hosted runner**. The input is hidden and is not
saved to disk or shell history.

Useful commands:

```bash
docker compose ps
docker compose logs --follow runner
docker compose stop
docker compose start
```

To target another repository or choose another runner name:

```bash
RUNNER_REPOSITORY=OWNER/REPO RUNNER_NAME=my-gpu-runner ./start.sh
```

## Security

This repository is public, so the GPU workflow intentionally does not run for
pull requests. A workflow running on a self-hosted runner can execute arbitrary
commands on that runner. Keep the workflow limited to trusted pushes and manual
dispatches, and do not add `pull_request_target` or fork-triggered jobs.

Stopping the container leaves the runner registered but offline. Before deleting
the named volumes, remove the runner in GitHub under **Settings → Actions →
Runners**, then run `docker compose down --volumes`.
