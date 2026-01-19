import os
import json
from functools import partial
from typing import Callable
from pathlib import Path
from .utils import timerPrint, get_file_path_str

# use `from functools import partial` to bind args:
# s3_download_file = partial(s3_download, s3_client, bucket_name)
def s3_download(s3_client, bucket_name: str, local_file_path: Path | str, s3_path: str):
  timerPrint(f'Downloading from S3: {s3_path}')
  s3_client.download_file(bucket_name, s3_path, get_file_path_str(local_file_path))

def s3_upload(s3_client, bucket_name: str, local_file_path: Path | str, s3_path: str):
  timerPrint(f'Uploading to S3: {s3_path}')
  s3_client.upload_file(get_file_path_str(local_file_path), bucket_name, s3_path)

def get_s3_downloader(s3_client, bucket_name: str) -> Callable[[Path | str, str], None]:
  return partial(s3_download, s3_client, bucket_name)

def get_s3_uploader(s3_client, bucket_name: str) -> Callable[[Path | str, str], None]:
  return partial(s3_upload, s3_client, bucket_name)

def download_s3_dir(s3_client, bucket_name: str, prefix: str, download_dir: Path):
  # List and download all objects with the specified key prefix
  # s3_client = boto_session.client('s3')
  paginator = s3_client.get_paginator('list_objects_v2')
  for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page.get('Contents', []):
      key = obj['Key']
      local_file_path = os.path.join(download_dir, os.path.basename(key))
      s3_client.download_file(bucket_name, key, local_file_path)

def get_secret_dict(boto_session, secret_name: str, region_name = "us-east-1") -> dict:
  '''
  sample code from aws console secrets manager
  https://docs.aws.amazon.com/secretsmanager/latest/userguide/retrieving-secrets-python-sdk.html
  boto_session = boto3.Session(region_name='us-east-1')
  may want to refactor this to conditionally parse json
  '''
  client = boto_session.client(service_name='secretsmanager', region_name=region_name)

  # For a list of exceptions thrown, see
  # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
  get_secret_value_response = client.get_secret_value(SecretId=secret_name)
  return json.loads(get_secret_value_response['SecretString'])
