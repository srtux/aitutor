"""AiTutor backend package for local multimodal tutoring."""

from .affect_router import AffectRouter, clamp, extract_euler_angles

__all__ = [
    "AffectRouter",
    "clamp",
    "extract_euler_angles",
]

try:
    from .curriculum import (
        generate_offline_socratic_response,
        get_initial_whiteboard,
        get_opening_greeting,
        get_proactive_prompt,
        get_topic,
        list_topics,
    )

    __all__.extend(
        [
            "generate_offline_socratic_response",
            "get_initial_whiteboard",
            "get_opening_greeting",
            "get_proactive_prompt",
            "get_topic",
            "list_topics",
        ]
    )
except ImportError:
    pass

try:
    from .stt_engine import LocalSTTEngine

    __all__.append("LocalSTTEngine")
except ImportError:
    pass

try:
    from .llm_client import LocalOllamaClient

    __all__.append("LocalOllamaClient")
except ImportError:
    pass

try:
    from .tts_engine import LocalTTSEngine

    __all__.append("LocalTTSEngine")
except ImportError:
    pass

try:
    from .main import app

    __all__.append("app")
except ImportError:
    pass
