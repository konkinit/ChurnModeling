import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class S3_configs:
    local_path: str = os.environ.get("LOCAL_PATH")
    endpoint: str = os.environ.get("ENDPOINT_URL")
    bucket: str = os.environ.get("BUCKET")
    path: str = os.environ.get("PATH")
    key_id: str = os.environ.get("KEY_ID")
    access_key: str = os.environ.get("ACCESS_KEY")
    token: str = os.environ.get("TOKEN")
