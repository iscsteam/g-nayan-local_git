import os 
import pytz
import psycopg2
from dotenv import load_dotenv


# Load env vars
load_dotenv()
def connection():
    # Get values
    host = os.getenv("host_env")
    db = os.getenv("dbname_env")
    user = os.getenv("user_env")
    password = os.getenv("password_env")
    sslrootcert = "root.crt"

    # Build the connection string
    connString = (
        f"host={host} port=5433 dbname={db} user={user} password={password} "
        f"sslmode=verify-ca sslrootcert={sslrootcert}"
    )

    # Connect
    try:
        conn = psycopg2.connect(connString)
        print("✅ Connected successfully!")
        return conn
    except Exception as e:
        print("❌ Connection failed:")
        print(e)


