IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

RETRIEVAL_TEXT = "You are a helpful assistant. Here, you act as an LLM Retriever. Retrieve an image-description pair that serves as evidence to answer the question related to this image, and generate a [RET] token for retrieval."
CAND_TEXT = "You are a helpful assistant. Here, you act as an LLM Retriever. Summarize the provided image and text candidates, and generate a [RET] token for retrieval."

IN_ONE_WORD_I = "Summarize above image in one word:"
IN_ONE_WORD_T = "Summarize above text in one word:"
IN_ONE_WORD_MM = "Summarize above image and text in one word:"

# RETRIEVAL_TEXT = "You are a helpful assistant. Here, give you a query and you should generate a [RET] token for retrieval."
# CAND_TEXT = "You are a helpful assistant. Here, give you a corpus and you should generate a [RET] token for retrieval."