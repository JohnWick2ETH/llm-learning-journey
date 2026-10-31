# Local GPU GitHub Actions runner

This directory runs a repository-scoped GitHub Actions runner in Docker with
the host NVIDIA GPU passed through. The runner has the custom label `gpu`;
GitHub also gives it the standard `self-hosted`, `linux`, and `x64` labels.

## Prerequisites

- An NVIDIA driver and NVIDIA Container Toolkit on the host.
- Docker Engine with Compose 2.30 or newer.
- GitHub CLI (`gh`) authenticated as a repository administrator.

The container deliberately does not mount `/var/run/docker.sock`. Jobs can use
the GPU, network, and their workspace, but cannot control the host Docker daemon.

## Manage the runner

From this directory, run:

```bash
./runner_admin.sh start
```

`runner_admin.sh start` checks GPU passthrough, downloads and checksum-verifies the pinned
official Actions runner, builds its image, requests a one-hour registration
token through the authenticated `gh` CLI, and waits until GitHub reports that it
is listening for jobs. Runner state and job workspaces live in named Docker
volumes, so a normal container restart does not require registration again.

The admin commands are:

```bash
./runner_admin.sh status
./runner_admin.sh logs
./runner_admin.sh logs --follow
./runner_admin.sh deactivate
```

`status` reports whether the container is stopped, idle, or processing a job.
`logs` shows the last 100 lines by default and accepts Docker Compose log options.
`deactivate` refuses to interrupt an active job, stops the container, removes its
GitHub registration, and removes the stopped container while preserving the
image and named volumes.

Creating a registration token requires repository administration permission. If
the authenticated `gh` account lacks that permission, `runner_admin.sh` prompts for a
short-lived token. A repository administrator can obtain one under **Settings →
Actions → Runners → New self-hosted runner**. The input is hidden and is not
saved to disk or shell history.

To target another repository or choose another runner name:

```bash
RUNNER_REPOSITORY=OWNER/REPO RUNNER_NAME=my-gpu-runner ./runner_admin.sh start
```

### Local proxy bound to host loopback

If an HTTP or mixed proxy such as Clash Verge listens only on host
`127.0.0.1:1080`, start the runner with:

```bash
RUNNER_PROXY=http://127.0.0.1:1080 ./runner_admin.sh start
```

`runner_admin.sh` applies the proxy only to the runner container and its job
subprocesses. Its own downloads and the Docker build do not use this proxy. The
runner container uses host networking because container loopback cannot reach
host loopback through Docker's normal bridge network. Proxy variables are
provided in lowercase for the Actions runner and uppercase for tools that
expect those names.

Host networking lets the runner access every service bound to the host network,
not only the proxy. Only enable `RUNNER_PROXY` for trusted workflows. To exclude
additional hostnames from the proxy, set a comma-separated hostname list:

```bash
RUNNER_PROXY=http://127.0.0.1:1080 \
RUNNER_NO_PROXY=localhost,127.0.0.1,internal.example.com \
./runner_admin.sh start
```

## Security

This repository is public, so the GPU workflow intentionally does not run for
pull requests. A workflow running on a self-hosted runner can execute arbitrary
commands on that runner. Keep the workflow limited to trusted pushes and manual
dispatches, and do not add `pull_request_target` or fork-triggered jobs.

Deactivation uses GitHub's short-lived removal-token API. If the authenticated
`gh` account lacks repository administration permission, the script prompts for
a token from **Settings → Actions → Runners → cs336-gpu-runner → Remove**. The
input is hidden and is not saved to disk or shell history.

Deleting the named volumes is a separate destructive operation and is not part
of deactivation.
