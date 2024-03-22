import os 
import sys 
import logging

logging_str = "[%(asctime)s: %(levelname)s : %(module)s : %(message)s]"
#logging_str: This variable contains a format string for log messages
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")
os.makedirs(log_dir, exist_ok = True)

logging.basicConfig(
    level = logging.INFO,
    format = logging_str,

    handlers = [
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)

    ]
)

logger = logging.getLogger("cnnClassifierLogger")

'''Overall, this code sets up a 
logging system that logs messages 
to both a file and the console, using 
a specific format for the log messages.
 It also creates a logger object that you 
 can use to log messages in your code
'''