import requests
from collections.abc import Mapping


url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'city':1, 'type':1, 'bhk':2})

print(r.json())