import os
from pathlib import Path

def download_s3_dir(s3_client, bucket_name: str, prefix: str, download_dir: Path):
  # List and download all objects with the specified key prefix
  # s3_client = boto_session.client('s3')
  paginator = s3_client.get_paginator('list_objects_v2')
  for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page.get('Contents', []):
      key = obj['Key']
      local_file_path = os.path.join(download_dir, os.path.basename(key))
      s3_client.download_file(bucket_name, key, local_file_path)
