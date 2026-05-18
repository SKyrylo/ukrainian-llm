# Module responsible for setting up a consistent logger across the project.
# All modules that need logging should call setup_logger(__name__) from here
# rather than configuring logging independently, to keep output formatting uniform.
import logging

# setup_logger returns a named logger with a StreamHandler that prints
# timestamped, leveled messages to stderr. The guard on logger.handlers
# prevents duplicate log lines when the same module is imported multiple times
# (e.g. in a Jupyter notebook that re-runs a cell).
def setup_logger(name):
    logger = logging.getLogger(name)
    # Only add handlers if they haven't been added yet (prevents duplicates)
    if not logger.handlers:
        # Set the minimum severity level that will be captured by this logger
        logger.setLevel(logging.INFO)
        # StreamHandler sends log records to sys.stderr by default
        handler = logging.StreamHandler()
        # Format: [timestamp] [module_name] [LEVEL] - message
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
