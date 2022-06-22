import s3fs
import pandas as pd

# enabling conexion to s3 storage for data retrieving

def import_from_S3(bucket, path, key_id, access_key, token):
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url':'https://'+'minio.lab.sspcloud.fr'}, key=key_id, secret=access_key, token=token)
    return pd.read_sas(fs.open(bucket+"/"+path+"/commsdata.sas7bdat"), format='sas7bdat')
