import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import (
    import_from_local,
    import_from_S3
)
from src.configs import ImportData


def import_data():
    params = ImportData()
    if os.path.isfile(os.path.join(params.local_path)):
        return import_from_local(params.local_path)
    return import_from_S3(
        params.endpoint,
        params.bucket,
        params.path,
        params.key_id,
        params.access_key,
        params.token
    )
