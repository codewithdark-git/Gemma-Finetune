import logging
import os
from datetime import datetime

class LogManager:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # Configure logging
        self.logger = logging.getLogger("GemmaFinetuning")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.logger.addHandler(file_handler)
    
    def log(self, message, level="info"):
        """Log a message with specified level"""
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        
    def get_current_log_file(self):
        """Get path to current log file"""
        return self.log_file
    
    def read_logs(self):
        """Read current log file contents"""
        try:
            with open(self.log_file, 'r') as f:
                return f.read()
        except Exception:
            return "Error reading log file"
