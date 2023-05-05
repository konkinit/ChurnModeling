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
        _df = import_from_local(params.local_path)
    _df = import_from_S3(
        params.endpoint,
        params.bucket,
        params.path,
        params.key_id,
        params.access_key,
        params.token
    )
    return _df.apply(
            lambda x: x.apply(
                lambda z: z.decode("utf-8") if type(z) == bytes else z),
            axis=1)
