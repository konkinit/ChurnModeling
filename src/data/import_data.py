import s3fs
from pandas import read_csv, read_sas

def import_from_S3(
        bucket, 
        path, 
        key_id, 
        access_key, 
        token):
    """
    enabling conexion to s3 storage for data retrieving
    """
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url':'https://'+'minio.lab.sspcloud.fr'},
        key=key_id,
        secret=access_key,
        token=token)

    return read_sas(
        fs.open(f"{bucket}/{path}/commsdata.sas7bdat"),
        format='sas7bdat')


def import_from_local(path):
    return read_csv(f"{path}/data/raw/commsdata.csv", sep="|")
