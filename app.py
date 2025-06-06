from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import base64
import cv2
import numpy as np
from datetime import datetime
import csv
import requests
import shutil

app = Flask(__name__)
UPLOAD_FOLDER = 'dataset'
CSV_LOG = 'metadata.csv'
BOT_TOKEN = '8155990401:AAF1F2S7nlg2aUEteB8enKIE2h57PyrxUdA'  
CHAT_ID = '8073724693'      
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def send_zip_to_telegram(person):
    zip_filename = f'/persistent/{person}.zip'
    folder_to_zip = os.path.join(UPLOAD_FOLDER, person)

    shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', folder_to_zip)

    with open(zip_filename, 'rb') as f:
        files = {'document': f}
        url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendDocument?chat_id={CHAT_ID}'
        r = requests.post(url, files=files)
        print("Telegram Response:", r.text)

@app.route('/upload', methods=['POST'])
def upload():
    data_url = request.form['image']
    expression = request.form['expression']
    person = request.form['person']

    img_data = base64.b64decode(data_url.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img = img[y:y+h, x:x+w]

    folder = os.path.join(UPLOAD_FOLDER, person, expression)
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f'{timestamp}.jpg'
    filepath = os.path.join(folder, filename)
    rel_path = f'dataset/{person}/{expression}/{filename}'
    cv2.imwrite(filepath, img)

    with open(CSV_LOG, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, person, expression, rel_path])

    expressions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    all_done = all(os.path.exists(os.path.join(UPLOAD_FOLDER, person, exp)) and 
                   len(os.listdir(os.path.join(UPLOAD_FOLDER, person, exp))) > 0
                   for exp in expressions)

    if all_done:
        send_zip_to_telegram(person)

    return 'Image saved'

@app.route('/images/<person>/<expression>')
def list_images(person, expression):
    folder = os.path.join(UPLOAD_FOLDER, person, expression)
    if not os.path.exists(folder):
        return jsonify([])
    files = os.listdir(folder)
    files.sort(reverse=True)
    return jsonify([f'/static_images/{person}/{expression}/{file}' for file in files])

@app.route('/static_images/<person>/<expression>/<filename>')
def serve_image(person, expression, filename):
    folder = os.path.join(UPLOAD_FOLDER, person, expression)
    return send_from_directory(folder, filename)

@app.route('/download_csv')
def download_csv():
    return send_from_directory('/persistent', 'metadata.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=10000)
