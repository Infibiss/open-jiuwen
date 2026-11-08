# openJiuwen budget-aware harness
FROM python:3.12.7-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/tmp/tiktoken \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/tmp

WORKDIR /app

# all versions pinned: requirements.txt (top level) + constraints.txt (full transitive lock)
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt \
 && python -m compileall -q /usr/local/lib/python3.12/site-packages > /dev/null 2>&1 || true

COPY harness ./harness
COPY tools ./tools
COPY run_agent.py README.md ./

# precompile (the container runs with --read-only) and smoke-test the imports offline
RUN python -m compileall -q /app \
 && python -c "import harness.oj, harness.scheduler, harness.react, harness.llm; print('imports ok')"

ENTRYPOINT ["python", "/app/run_agent.py"]
