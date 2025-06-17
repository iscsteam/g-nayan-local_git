import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
import torch
from pydantic_settings import BaseSettings
from pydantic import BaseModel

load_dotenv()

# Old connection function (to be deprecated/removed)
def connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST", "mysql"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "iscs"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DB")
        )
        # print("✅ Connected to MySQL (old method)") # Silencing print for cleaner output
        return conn
    except Error as e:
        # print(f"❌ MySQL connection error (old method): {e}") # Silencing print
        return None

class DatabaseSettings(BaseModel):
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "mysql")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "iscs")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "") # Provide default for getenv
    MYSQL_DB: str = os.getenv("MYSQL_DB", "") # Provide default for getenv

class AppSettings(BaseSettings):
    DB: DatabaseSettings = DatabaseSettings()

    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "minio:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
    MINIO_BUCKET_NAME: str = os.getenv("MINIO_BUCKET_NAME", "fundus-images")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "False").lower() == "true"

    FASTAPI_SECRET_KEY: str = os.getenv("FASTAPI_SECRET_KEY", "please_change_this_secret_in_env_very_important")

    RESNET18_PATH: str = os.getenv("RESNET18_PATH", "model_paths/ResNet18.pth")
    EFFINET_MODEL_V2_PATH: str = os.getenv("EFFINET_MODEL_V2_PATH", "model_paths/effinet_model_V2.pth")
    LEFT_RIGHT_MODEL_PATH: str = os.getenv("LEFT_RIGHT_MODEL_PATH", "model_paths/left_right_Model.pth")

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        env_file = ".env"
        extra = "ignore" # Allow other env vars not defined in model

_app_settings_instance = None

def get_app_settings() -> AppSettings:
    global _app_settings_instance
    if _app_settings_instance is None:
        _app_settings_instance = AppSettings()
    return _app_settings_instance
