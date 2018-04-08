import requests
url = 'http://127.0.0.1:5000/obj_detection_api'
files = {'file': open('/home/frankle/Documents/Data Science Intro/ObjectDetection/test_images/image1.jpg', 'rb')}
r = requests.post(url, files=files)
print(r.json())