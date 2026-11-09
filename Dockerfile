# openJiuwen budget-aware harness
FROM python:3.12.7-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_NO_CACHE=1 \
    UV_HTTP_TIMEOUT=120 \
    TIKTOKEN_CACHE_DIR=/tmp/tiktoken \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/tmp

WORKDIR /app

# All versions pinned: requirements.txt (top level) + constraints.txt (full transitive lock).
# uv downloads the ~170 wheels in parallel (pip fetches them one by one); if uv is
# unavailable for any reason the same pinned set is installed with pip.
COPY requirements.txt constraints.txt ./
RUN (pip install uv==0.5.11 \
     && uv pip install --system -r requirements.txt -c constraints.txt \
     && pip uninstall -y uv) \
 || pip install --no-compile -r requirements.txt -c constraints.txt

COPY harness ./harness
COPY tools ./tools
COPY run_agent.py README.md ./

# The container runs with --read-only: write .pyc only for the modules the harness
# actually imports (much faster than compiling all of site-packages), then smoke-test.
RUN PYTHONDONTWRITEBYTECODE= python -c "import harness.oj, harness.scheduler, harness.react, harness.llm, harness.llm_solvers; print('imports ok')" \
 && python -m compileall -q /app

ENTRYPOINT ["python", "/app/run_agent.py"]
