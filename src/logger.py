import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    # Only add handlers if they haven't been added yet (prevents duplicates)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
