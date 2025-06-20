import logging
from datetime import datetime
from typing import Optional
from mysql.connector.abstracts import MySQLConnectionAbstract, MySQLCursorAbstract
# To get DB connection in a way that's compatible with the new structure
from src.services.database import get_db_connection, DBError # Assuming DBError is mysql.connector.Error

# Global flag to ensure setup_logging is called only once
_logging_configured = False

class MySQLHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.conn: Optional[MySQLConnectionAbstract] = None
        self.cursor: Optional[MySQLCursorAbstract] = None
        # Defer connection acquisition to the first emit, or manage externally
        # For now, let's try to connect on init but handle failure gracefully.
        try:
            self._connect()
        except Exception as e:
            # Using print here as logger itself is not yet configured
            print(f"MySQLHandler: Failed to connect during init: {e}. Will attempt reconnect on emit.") # Keep this print for critical init failure

    def _connect(self) -> None:
        # Close existing connection if any
        self._close_conn()
        try:
            self.conn = get_db_connection() # Uses the refactored DB connection logic
            if self.conn:
                self.cursor = self.conn.cursor()
        except DBError: # Keep 'e' if you log it, otherwise it's unused. Removed print, so e is unused.
            self.conn = None # Ensure conn is None if connection fails
            self.cursor = None
            # Not raising error here, will try to reconnect or skip logging in emit

    def _close_conn(self) -> None:
        if self.cursor:
            try:
                self.cursor.close()
            except Exception: # pragma: no cover
                pass # Ignore errors on close
            self.cursor = None
        if self.conn:
            try:
                self.conn.close()
            except Exception: # pragma: no cover
                pass # Ignore errors on close
            self.conn = None

    def emit(self, record: logging.LogRecord) -> None:
        # Try to connect if not connected
        if not self.conn or not self.cursor:
            try:
                self._connect()
            except Exception: # pragma: no cover; Removed 'e' as it's not used, and removed print
                return # Skip logging if can't connect

        if not self.conn or not self.cursor: # Still no connection
             return


        log_time = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        level = record.levelname
        msg = self.format(record)

        try:
            self.cursor.execute(
                "INSERT INTO api_logs (timestamp, level, message) VALUES (%s, %s, %s)",
                (log_time, level, msg)
            )
            self.conn.commit()
        except DBError: # Keep 'e' if you log it. Removed print, so 'e' is unused.
            # Attempt to reconnect and retry once
            try:
                self._connect() # Reconnect
                if self.conn and self.cursor: # Check added for self.cursor as well
                    self.cursor.execute(
                        "INSERT INTO api_logs (timestamp, level, message) VALUES (%s, %s, %s)",
                        (log_time, level, msg)
                    )
                    self.conn.commit()
                else: # pragma: no cover
                    pass
            except Exception: # pragma: no cover; Removed 'retry_e' as it's not used, and removed print
                # Consider logging this failure to a fallback (e.g., file or console)
                pass # Skip if retry fails
        except Exception: # pragma: no cover; Removed 'e' as it's not used, and removed print
            pass


    def close(self) -> None:
        self._close_conn()
        super().close()

def setup_logging() -> None:
    # This function should be called once, e.g., in lifespan startup.
    global _logging_configured
    if _logging_configured:
        return

    # Get the root logger
    # Configuring the root logger will apply to all loggers obtained via logging.getLogger()
    # unless they have specific propagation settings.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO) # Set root level, handlers can have their own levels

    # Clear existing handlers on the root logger, if any, to avoid duplicates if setup_logging is called multiple times by mistake
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console handler (good for seeing logs during development and in container logs)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO) # Or DEBUG
    root_logger.addHandler(console_handler)

    # File handler
    try:
        file_handler = logging.FileHandler("api_hits.log", mode='a') # Append mode
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
    except Exception as e: # pragma: no cover
        root_logger.error(f"Failed to initialize file logger: {e}", exc_info=True)


    # MySQL handler (conditionally add if DB logging is desired and works)
    # Be cautious with this in production due to performance.
    try:
        db_handler = MySQLHandler()
        db_handler.setFormatter(formatter) # You might want a different format for DB
        db_handler.setLevel(logging.INFO) # Log INFO and above to DB
        root_logger.addHandler(db_handler)
    except Exception as e: # pragma: no cover
        root_logger.error(f"Failed to initialize MySQL logging handler: {e}", exc_info=True)

    _logging_configured = True
    root_logger.info("Logging configured (console, file, and potentially MySQL).")

# Example of how other modules should get a logger:
# import logging
# logger = logging.getLogger(__name__)
# logger.info("This is a test log message.")

# If you want a specific named logger that's not root, but inherits root config:
# def get_custom_logger(name: str) -> logging.Logger:
#    return logging.getLogger(name)
