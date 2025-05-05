import os 
from dotenv import load_dotenv
load_dotenv()
import mysql.connector
from mysql.connector import Error
import os

def connection():
    try:
        connection = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "mysql"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "iscs"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB")
        )
        print("✅ Connected to MySQL")
        return connection
    except Error as e:
        print("❌ MySQL connection error:", e)
        return None





