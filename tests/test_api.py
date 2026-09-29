from fastapi.testclient import TestClient
from app import app
client=TestClient(app)
def test_health():
    r=client.get('/health');assert r.status_code==200;assert r.json()['status']=='ok'
def test_nutrition():
    r=client.post('/api/nutrition',json=[{'name':'steamed white rice','grams':200}]);assert r.status_code==200;assert r.json()['totals']['kcal']==260
def test_bad_image():
    r=client.post('/api/analyze',files={'file':('bad.txt',b'not an image','text/plain')});assert r.status_code==415
