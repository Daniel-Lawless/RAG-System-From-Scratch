import logging

def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)

    noisy_loggers = [
        "httpx",
        "httpcore",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
        "openai",
        "openai._base_client",
        "vector_search",
        "keyword_search",
        "hybrid_search",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)