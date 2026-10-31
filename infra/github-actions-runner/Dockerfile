FROM ubuntu:24.04

ARG RUNNER_VERSION=2.338.2
ARG RUNNER_SHA256=baebf204307bac6193480852732f0aba2ae97b6c5597c244f4d5558390147c40

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ca-certificates \
        curl \
        git \
        gosu \
        jq \
        libicu74 \
        libssl3 \
        python3 \
        python3-venv \
        zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1001 --shell /bin/bash runner \
    && mkdir -p /opt/actions-runner /home/runner/actions-runner /home/runner/_work

COPY .cache/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz /tmp/actions-runner.tar.gz

RUN echo "${RUNNER_SHA256}  /tmp/actions-runner.tar.gz" | sha256sum --check --strict \
    && tar --extract --gzip --file /tmp/actions-runner.tar.gz --directory /opt/actions-runner \
    && rm /tmp/actions-runner.tar.gz \
    && chown -R runner:runner /opt/actions-runner /home/runner

COPY --chmod=755 entrypoint.sh /usr/local/bin/runner-entrypoint

WORKDIR /home/runner/actions-runner
ENTRYPOINT ["/usr/local/bin/runner-entrypoint"]
