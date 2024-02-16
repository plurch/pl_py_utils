import requests
import json

def post_request(url: str, data_dict: dict) -> requests.Response:
  """
  Send a POST request

  Check status code: `response.status_code`
  Get JSON value: `response.json()`
  """
  json_data = json.dumps(data_dict)
  return requests.post(url, data=json_data, headers={"Content-Type": "application/json"})
