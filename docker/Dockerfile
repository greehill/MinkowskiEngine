# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa AS uv

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG PYTHON_VERSION=3.12.13
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0+PTX"

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV MAX_JOBS=2
ENV MINKOWSKI_BLAS=openblas
ENV MINKOWSKI_FORCE_CUDA=1
ENV PATH="/opt/venv/bin:/usr/local/cuda/bin:${PATH}"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV UV_LINK_MODE=copy
ENV UV_NO_PROGRESS=1
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV VIRTUAL_ENV=/opt/venv

RUN --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    libopenblas-dev \
    ninja-build

COPY --from=uv /uv /uvx /usr/local/bin/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv python install ${PYTHON_VERSION} && \
    uv venv --python ${PYTHON_VERSION} ${VIRTUAL_ENV}
ENV UV_PYTHON_DOWNLOADS=never

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install \
    setuptools==80.10.2 \
    wheel \
    packaging \
    ninja \
    numpy==2.2.6 \
    "torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=628e89bd5110ced7debee2a57c69959725b7fbc64eab81a39dd70e46c7e28ba5"

WORKDIR /opt/MinkowskiEngine
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv pip install --no-build-isolation --no-deps . && \
    uv pip check

# CUDA 12 minor-version compatibility uses the host driver on driver 525+.
# Do not let the base image's forward-compatibility libcuda shadow that driver.
RUN rm -rf /usr/local/cuda/compat

CMD ["python", "-c", "import MinkowskiEngine as ME; import torch; print(ME.__version__, torch.__version__, torch.version.cuda)"]
