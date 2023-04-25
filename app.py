from flask import Flask, render_template, Response, jsonify
from algorithm.object_detector import YOLOv7
from utils.detections import draw
import cv2
import urllib.request
import numpy as np

app = Flask(__name__)

# Load the YOLOv7 object detector and set the OCR classes
yolov7 = YOLOv7()
yolov7.set(ocr_classes=['number_plate'])
yolov7.load('mybest.weights', classes='classes.yaml', ocr_weights='last', device='cpu') # use 'gpu' for CUDA GPU inference

# Connect to the ESP32 cam's image stream
url = "http://192.168.1.13/cam-hi.jpg"

def gen_frames():
    while True:
        # Retrieve the image from the specified URL
        img_resp = urllib.request.urlopen(url)
        img_arr = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        # Perform license plate detection on the image
        detections = yolov7.detect(img)
        detected_image = draw(img, detections)

        # Convert the image to JPEG format and yield it to the web page
        ret, buffer = cv2.imencode('.jpg', detected_image)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('client.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
