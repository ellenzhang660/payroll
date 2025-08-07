import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Prevent duplicate logging via propagation to ancestor loggers (like root)
logger.propagate = False

# Check if handlers already exist to avoid adding multiple
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
