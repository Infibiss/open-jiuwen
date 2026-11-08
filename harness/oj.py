"""Single import point for openJiuwen.

The SDK logs INFO to console and to ./logs/* by default; the container's
filesystem is read-only except /tmp, so logging is reconfigured *before* any
other openJiuwen module is imported.
"""
import os

os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/tmp/tiktoken")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if not os.environ.get("SSL_CERT_FILE"):
    # python.org macOS builds ship without a CA bundle; Debian images have one.
    import ssl
    if not os.path.isfile(ssl.get_default_verify_paths().openssl_cafile or ""):
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass

from openjiuwen.core.common.logging.default.constant import DEFAULT_INNER_LOG_CONFIG  # noqa: E402
from openjiuwen.core.common.logging.log_config import configure_log_config  # noqa: E402

configure_log_config({
    **DEFAULT_INNER_LOG_CONFIG,
    "level": "CRITICAL",
    "output": ["console"],
    "interface_output": ["console"],
    "performance_output": ["console"],
})

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig  # noqa: E402,F401
from openjiuwen.core.foundation.tool import tool  # noqa: E402,F401
from openjiuwen.core.single_agent import AgentCard, ReActAgent, ReActAgentConfig  # noqa: E402,F401
from openjiuwen.core.single_agent.rail.base import AgentCallbackEvent  # noqa: E402,F401

__all__ = ["Model", "ModelClientConfig", "ModelRequestConfig", "tool", "AgentCard", "ReActAgent",
           "ReActAgentConfig", "AgentCallbackEvent"]
