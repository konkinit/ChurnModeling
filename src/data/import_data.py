import pandas as pd
import numpy as np
import yaml
import s3fs

# enabling conexion to s3 storage for data retrieving
def import_data():
    credentials = yaml.safe_load(open('/home/coder/.config/code-server/telco_churn/config.yaml', 'rb').read())
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'}, 
        key=credentials['input']['key_id'], secret=credentials['input']['access_key'], 
        token=credentials['input']['token'])
    return pd.read_sas(fs.open("ikonkobo/commsdata/commsdata.sas7bdat"), format='sas7bdat')