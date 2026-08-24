# logger.py

from datetime import datetime

class Logger:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def log(self, message):
        """Debug logging - only shown in verbose mode"""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {message}")

    def always_log(self, message):
        """Always log to console"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {message}")

    def set_verbose(self, verbose):
        self.verbose = verbose

# Global logger instance
logger = Logger()