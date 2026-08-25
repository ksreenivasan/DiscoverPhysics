FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    PYTHONPATH=/opt/discoverphysics/PhysicsSchool:/opt/discoverphysics/ScienceAgent

RUN useradd --create-home --uid 10001 evaluser \
    && mkdir -p /ipc /artifacts \
    && chown -R evaluser:evaluser /ipc /artifacts

WORKDIR /app
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt

# This fork contains the pinned canonical source in-tree; no nested checkout.
COPY PhysicsSchool /opt/discoverphysics/PhysicsSchool
COPY ScienceAgent /opt/discoverphysics/ScienceAgent
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
COPY tests /app/tests
COPY configs /app/configs
RUN python -m pip install --no-cache-dir --no-deps /app

USER evaluser
WORKDIR /opt/discoverphysics
ENTRYPOINT ["dp-eval"]
