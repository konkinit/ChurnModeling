import json
from dataclasses import dataclass


with open("./data/tokens/tokens.json") as f:
    tokens = json.load(f)


@dataclass
class ImportData:
    local_path: str = "./data/commsdata.sasb7dat"
    endpoint: str = tokens["endpoint_url"]
    bucket: str = tokens["bucket"]
    path: str = tokens["path"]
    key_id: str = tokens["key_id"]
    access_key: str = tokens["access_key"]
    token: str = tokens["token"]
