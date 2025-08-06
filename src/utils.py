import logging

# Create a logger
logger = logging.getLogger(__name__)  # __name__ is the module name

# Set the minimum logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

# Create a console handler to output logs to the terminal
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)  # Set level for this handler

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

