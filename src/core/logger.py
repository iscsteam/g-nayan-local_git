import logging
from datetime import datetime
# To get DB connection in a way that's compatible with the new structure
from src.services.database import get_db_connection, DBError # Assuming DBError is mysql.connector.Error

# Global flag to ensure setup_logging is called only once
_logging_configured = False

class MySQLHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.conn = None
        self.cursor = None
        # Defer connection acquisition to the first emit, or manage externally
        # For now, let's try to connect on init but handle failure gracefully.
        try:
            self._connect()
        except Exception as e:
            # Using print here as logger itself is not yet configured
            print(f"MySQLHandler: Failed to connect during init: {e}. Will attempt reconnect on emit.")

    def _connect(self):
        # Close existing connection if any
        self._close_conn()
        try:
            self.conn = get_db_connection() # Uses the refactored DB connection logic
            if self.conn:
                self.cursor = self.conn.cursor()
                # print("MySQLHandler: Successfully connected to DB.") # Debug
        except DBError as e:
            self.conn = None # Ensure conn is None if connection fails
            self.cursor = None
            # print(f"MySQLHandler: Error connecting to DB: {e}") # Debug
            # Not raising error here, will try to reconnect or skip logging in emit

    def _close_conn(self):
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

    def emit(self, record):
        # Try to connect if not connected
        if not self.conn or not self.cursor:
            try:
                self._connect()
            except Exception as e: # pragma: no cover
                # print(f"MySQLHandler: Failed to connect on emit: {e}. Log will be skipped.") # Debug
                return # Skip logging if can't connect

        if not self.conn or not self.cursor: # Still no connection
             # print("MySQLHandler: No DB connection, skipping log.") # Debug
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
        except DBError as e: # Catch specific DB errors
            # print(f"MySQLHandler: Failed to log to DB: {e}. Attempting to reconnect.") # Debug
            # Attempt to reconnect and retry once
            try:
                self._connect() # Reconnect
                if self.conn and self.cursor:
                    self.cursor.execute(
                        "INSERT INTO api_logs (timestamp, level, message) VALUES (%s, %s, %s)",
                        (log_time, level, msg)
                    )
                    self.conn.commit()
                else: # pragma: no cover
                    # print("MySQLHandler: Reconnect failed, log skipped.") # Debug
                    pass
            except Exception as retry_e: # pragma: no cover
                # print(f"MySQLHandler: Failed to log to DB after retry: {retry_e}") # Debug
                # Consider logging this failure to a fallback (e.g., file or console)
                pass # Skip if retry fails
        except Exception as e: # Catch any other unexpected errors # pragma: no cover
            # print(f"MySQLHandler: Unexpected error during emit: {e}") # Debug
            pass


    def close(self):
        self._close_conn()
        super().close()

def setup_logging():
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
        # print("MySQL logging handler configured.") # Debug
    except Exception as e: # pragma: no cover
        root_logger.error(f"Failed to initialize MySQL logging handler: {e}", exc_info=True)
        # print(f"Failed to initialize MySQL logging handler: {e}") # Debug

    _logging_configured = True
    root_logger.info("Logging configured (console, file, and potentially MySQL).")

# Example of how other modules should get a logger:
# import logging
# logger = logging.getLogger(__name__)
# logger.info("This is a test log message.")

# If you want a specific named logger that's not root, but inherits root config:
# def get_custom_logger(name: str):
#    return logging.getLogger(name)
