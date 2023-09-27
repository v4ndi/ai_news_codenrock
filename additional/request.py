import requests
import json
import pandas as pd

url = "http://localhost:8000/predict/"
files = {"file": ("for_testing.csv", open("for_testing.csv", "rb"), "text/csv")}
response = requests.post(url, files=files)

if response.status_code == 200:
    data = json.loads(response.content)
    data_df = pd.DataFrame.from_dict(data)
    data_df.to_csv('result_api_docker.csv', index=False, encoding='utf-8-sig')

else:
    print(f"Error: {response.status_code}, {response.text}")