import json
import urllib.request

URL = "http://127.0.0.1:5000/predict"

sample_data = [
    {
        "university": "Banaras Hindu University",
        "year": 2024,
        "programme": "2-year Diploma in Pharmacy",
        "college": "Acharya Narendra Dev College",
        "variant": "Accounting and Finance + Economics",
        "round": "1",
        "category": "CW",
        "seat_type": "Paid Fee Course"
    }
]

req = urllib.request.Request(
    URL,
    data=json.dumps(sample_data).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

try:
    with urllib.request.urlopen(req) as response:
        res_body = response.read().decode("utf-8")
        print("Status Code:", response.status)
        print("Response:", res_body)
except Exception as e:
    print(f"Failed to query {URL}: {e}")
    print("Ensure the server is running with 'python app.py' first.")
