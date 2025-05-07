# logger.py

import logging
from datetime import datetime
from config import connection  # Import the connection function

class MySQLHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.conn = connection()  # Get connection from your function
        if self.conn:
            self.cursor = self.conn.cursor()

    def emit(self, record):
        log_time = datetime.fromtimestamp(record.created)
        level = record.levelname
        msg = self.format(record)

        try:
            if self.conn:
                self.cursor.execute(
                    "INSERT INTO api_logs (timestamp, level, message) VALUES (%s, %s, %s)",
                    (log_time, level, msg)
                )
                self.conn.commit()
        except Exception as e:
            print(f"Failed to log to DB: {e}")

    def close(self):
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if self.conn:
            self.conn.close()
        super().close()


def get_logger(name="my_app_logger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = logging.FileHandler("api_hits.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # MySQL handler
        db_handler = MySQLHandler()
        db_handler.setFormatter(formatter)
        logger.addHandler(db_handler)

    return logger
