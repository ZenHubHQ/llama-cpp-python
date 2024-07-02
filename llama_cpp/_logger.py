import sys
import ctypes
import logging

import llama_cpp

# enum ggml_log_level {
#     GGML_LOG_LEVEL_ERROR = 2,
#     GGML_LOG_LEVEL_WARN = 3,
#     GGML_LOG_LEVEL_INFO = 4,
#     GGML_LOG_LEVEL_DEBUG = 5
# };
GGML_LOG_LEVEL_TO_LOGGING_LEVEL = {
    2: logging.ERROR,
    3: logging.WARNING,
    4: logging.INFO,
    5: logging.DEBUG,
}

# Set up logger with custom handler
logger = logging.getLogger("llama-cpp-python")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@llama_cpp.llama_log_callback
def llama_log_callback(
    level: int,
    text: bytes,
    user_data: ctypes.c_void_p,
):
    if logger.level <= GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level]:
        _text = text.decode("utf-8")
        if _text.endswith("\n"):
            _text = _text[:-1]
        
        # Skip if the message only contains "."
        if not _text == ".":
            logger.log(GGML_LOG_LEVEL_TO_LOGGING_LEVEL[level], _text)


llama_cpp.llama_log_set(llama_log_callback, ctypes.c_void_p(0))


def set_verbose(verbose: bool):
    logger.setLevel(logging.DEBUG if verbose else logging.ERROR)
