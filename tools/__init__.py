"""
HOSER Tools Package

Utilities and scripts for HOSER trajectory generation and analysis.
"""

from tools.model_detection import (
    extract_model_name,
    get_display_name,
    get_model_color,
    get_model_line_style,
    parse_model_components,
    detect_model_files,
    ModelFile,
    MODEL_PATTERNS,
    DISPLAY_NAMES,
    MODEL_COLORS,
    MODEL_LINE_STYLES,
)

__all__ = [
    "extract_model_name",
    "get_display_name",
    "get_model_color",
    "get_model_line_style",
    "parse_model_components",
    "detect_model_files",
    "ModelFile",
    "MODEL_PATTERNS",
    "DISPLAY_NAMES",
    "MODEL_COLORS",
    "MODEL_LINE_STYLES",
]
