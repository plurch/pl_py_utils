from typing import TypedDict

class CurrentRun(TypedDict):
  date: str
  bucket: str
  s3Key: str
  s3URI: str
