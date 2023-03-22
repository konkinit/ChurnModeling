import s3fs
from pandas import read_sas, DataFrame


def import_from_S3(
        endpoint: str,
        bucket: str,
        path: str,
        key_id: str,
        access_key: str,
        token: str) -> DataFrame:
    """
    enabling conexion to s3 storage for data retrieving
    """
    fs = s3fs.S3FileSystem(
            client_kwargs={'endpoint_url': endpoint},
            key=key_id,
            secret=access_key,
            token=token)

    return read_sas(
                fs.open(f"{bucket}/{path}/commsdata.sas7bdat"),
                format='sas7bdat')


def import_from_local(path) -> DataFrame:
    return read_sas(
                f"{path}/inputs/raw/commsdata.sas7bdat",
                encoding="utf-8").set_index("Customer_ID")
