import mysql.connector
from mysql.connector import pooling # Import the pooling module
from mysql.connector import Error as DBError
from typing import List, Any, Optional, Dict
import pytz
from datetime import datetime
from src.core.config import get_app_settings

# Global variable for the connection pool
db_pool: Optional[pooling.MySQLConnectionPool] = None # Type hint for clarity

def init_db_pool(config: Dict[str, Any], pool_size: int = 5, pool_name: str = "mypool"):
    # This function will be called during app startup (lifespan event)
    # to initialize the actual connection pool.
    global db_pool
    if db_pool: # Avoid re-initializing if called multiple times
        print("Database pool already initialized.")
        return

    try:
        print(f"Initializing database pool '{pool_name}' with size {pool_size} and config: {config}")
        db_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name=pool_name,
            pool_size=pool_size,
            pool_reset_session=True, # Ensures session state is reset
            **config # host, port, user, password, database from AppSettings.DB
        )
        # Test the pool by getting a connection (optional, but good for startup check)
        # conn_test = db_pool.get_connection()
        # print("Successfully got a test connection from the pool.")
        # conn_test.close()
        print(f"Database pool '{pool_name}' initialized successfully.")
    except DBError as e:
        print(f"Failed to initialize database pool: {e}")
        db_pool = None # Ensure pool is None if initialization fails
        raise # Re-raise the error to potentially stop app startup if DB is critical
    except Exception as e: # Catch any other unexpected errors during pool creation
        print(f"An unexpected error occurred during database pool initialization: {e}")
        db_pool = None
        raise


def close_db_pool():
    # For mysql.connector.pooling, explicit closing of the pool isn't typically done
    # as connections are managed by daemon threads. However, if other pooling libraries
    # were used, a close method would be relevant here.
    # This function is provided for completeness or future adaptation.
    global db_pool
    if db_pool:
        print("Closing database pool (conceptually, as mysql.connector's pool is self-managing).")
        # If there were an explicit pool.close() method, it would be called here.
        db_pool = None


def get_db_connection():
    global db_pool
    if not db_pool:
        # This case should ideally not happen if lifespan initializes the pool correctly.
        # It indicates a problem with application startup or configuration.
        print("Database pool not initialized. Attempting a direct fallback connection (not recommended for production).")
        # Fallback to direct connection (from previous refactoring step, for resilience during dev)
        # This fallback should be removed once pooling is stable and mandatory.
        settings = get_app_settings()
        try:
            conn = mysql.connector.connect(
                host=settings.DB.MYSQL_HOST,
                port=settings.DB.MYSQL_PORT,
                user=settings.DB.MYSQL_USER,
                password=settings.DB.MYSQL_PASSWORD,
                database=settings.DB.MYSQL_DB
            )
            return conn
        except DBError as e:
            print(f"Database connection error (direct fallback): {e}")
            raise # Re-raise the exception to be handled by the caller

    # Get a connection from the initialized pool
    try:
        conn = db_pool.get_connection()
        # print("Connection obtained from pool.") # Debug
        return conn
    except DBError as e:
        print(f"Error getting connection from pool: {e}")
        # Handle pool exhaustion or other pool errors
        # For example, could implement retries or raise a specific service unavailable error
        raise Exception(f"Failed to get database connection from pool: {e}") # Raise a more generic server error


def data_from_db(query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    conn = None
    cursor = None
    try:
        conn = get_db_connection() # Gets connection from pool (or fallback)
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params)
        result = cursor.fetchall()
        return result
    except Exception as e: # Catching general Exception as get_db_connection might raise non-DBError too
        # Log the original error for debugging
        print(f"Error executing query '{query[:100]}...': {e}")
        raise # Re-raise the caught exception to be handled by API layer
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close() # Returns connection to the pool

def save_feedback_to_db(feedback_data: Dict[str, Any], patient_id: str, email_id: Optional[str]) -> None:
    left_eye = feedback_data.get("left_eye")
    right_eye = feedback_data.get("right_eye")

    if not left_eye or not right_eye:
        raise ValueError("Invalid feedback data: Missing 'left_eye' or 'right_eye' fields.")

    query = """
    INSERT INTO diabetic_retinopathy (
        Patient_ID, Predicted_Class, Stage, Confidence, Explanation, Note,
        Risk_Factor, Review, Feedback, Doctors_Diagnosis, email_id, timestamp
    ) VALUES (%(patient_id)s, %(predicted_class)s, %(stage)s, %(confidence)s, %(explanation)s, %(note)s,
              %(risk_factor)s, %(review)s, %(feedback)s, %(doctors_diagnosis)s, %(email_id)s, %(timestamp)s);
    """

    utc_now = datetime.now(pytz.utc)
    kolkata_tz = pytz.timezone('Asia/Kolkata')
    kolkata_now_str = utc_now.astimezone(kolkata_tz).strftime('%Y-%m-%d %H:%M:%S')

    records_to_insert = []
    for eye_identifier_suffix, eye_data_dict in [("_left", left_eye), ("_right", right_eye)]:
        record = {
            "patient_id": str(patient_id) + eye_identifier_suffix,
            "predicted_class": eye_data_dict["predicted_class"],
            "stage": eye_data_dict["Stage"],
            "confidence": eye_data_dict["confidence"],
            "explanation": eye_data_dict["explanation"],
            "note": eye_data_dict["Note"],
            "risk_factor": eye_data_dict["Risk_Factor"],
            "review": eye_data_dict.get("review"),
            "feedback": eye_data_dict.get("feedback"),
            "doctors_diagnosis": eye_data_dict.get("doctors_diagnosis"),
            "email_id": email_id,
            "timestamp": kolkata_now_str
        }
        records_to_insert.append(record)

    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        for record_params_dict in records_to_insert:
            cursor.execute(query, record_params_dict)

        conn.commit()
    except Exception as e: # Catching general Exception
        if conn:
            try:
                conn.rollback()
            except DBError as rb_err: # pragma: no cover (difficult to test rollback failure)
                 print(f"Error during rollback: {rb_err}")
        print(f"Database error during feedback submission for patient {patient_id}: {e}")
        raise # Re-raise to be caught by API layer
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close() # Returns connection to the pool
