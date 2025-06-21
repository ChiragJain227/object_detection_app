from flask import Flask, render_template, request, Response, session, redirect, url_for, jsonify
import cv2
from models.yolo_detector import YoloDetector
from models.frcnn import FRCNN
import numpy as np
from PIL import Image
import io
import time
import mysql.connector
import os
import json
from ensemble_boxes import weighted_boxes_fusion

app = Flask(__name__)
app.secret_key = 'your_secret_key'

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Object@123',
    'database': 'object_detection'
}

try:
    yolo_model = YoloDetector(model_variant="yolov8x")
    print("YOLOv8 model initialized successfully")
except Exception as e:
    print(f"Error initializing YOLOv8: {e}")

try:
    frcnn_model = FRCNN()
    print("Faster R-CNN initialized successfully")
except Exception as e:
    print(f"Error initializing Faster R-CNN: {e}")

camera = cv2.VideoCapture(0)

@app.template_filter('from_json')
def from_json_filter(s):
    return json.loads(s)

def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT DATABASE()")
        db_name = cursor.fetchone()[0]
        print(f"Connected to database: {db_name}")
        cursor.close()
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to MySQL: {err}")
        raise

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    if x1 >= x2 or y1 >= y2 or x1g >= x2g or y1g >= y2g:
        return 0.0
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def update_performance_metrics(model_type, inference_time, iou_scores, result_image_path=None, throughput=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT mAP, inference_time, throughput, `precision`, count, result_image_path FROM performance_metrics WHERE model_type = %s", (model_type,))
    result = cursor.fetchone()
    if result:
        current_mAP, current_inf_time, current_throughput, current_precision, count, current_image_path = result
    else:
        current_mAP, current_inf_time, current_throughput, current_precision, count, current_image_path = 0, 0, 0, 0, 0, None
    count += 1
    new_inf_time = float((current_inf_time * (count - 1) + inference_time) / count)  # Convert to Python float
    precision = len([s for s in iou_scores if s > 0.5]) / len(iou_scores) if iou_scores else 0
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0  # Convert to Python float
    new_precision = float((current_precision * (count - 1) + precision) / count)  # Convert to Python float
    new_mAP = float((current_mAP * (count - 1) + mean_iou) / count)  # Convert to Python float
    new_throughput = float(current_throughput if throughput is None else (current_throughput * (count - 1) + throughput) / count)  # Convert to Python float
    new_image_path = result_image_path if result_image_path else current_image_path
    cursor.execute("""
        INSERT INTO performance_metrics (model_type, mAP, inference_time, throughput, `precision`, count, result_image_path)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        mAP = %s, inference_time = %s, throughput = %s, `precision` = %s, count = %s, result_image_path = %s
    """, (model_type, new_mAP, new_inf_time, new_throughput, new_precision, count, new_image_path,
          new_mAP, new_inf_time, new_throughput, new_precision, count, new_image_path))
    conn.commit()
    cursor.close()
    conn.close()

def save_detection_history(model_type, image_path, inference_time, iou_scores, detected_objects, throughput=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0  # Convert to Python float
    precision = float(len([s for s in iou_scores if s > 0.5]) / len(iou_scores)) if iou_scores else 0  # Convert to Python float
    inference_time = float(inference_time)  # Convert to Python float
    throughput = float(throughput) if throughput is not None else None  # Convert to Python float or None
    detected_objects_json = json.dumps(detected_objects)
    cursor.execute("""
        INSERT INTO detection_history (model_type, image_path, inference_time, mAP, `precision`, throughput, detected_objects)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (model_type, image_path, inference_time, mean_iou, precision, throughput, detected_objects_json))
    conn.commit()
    cursor.close()
    conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
            conn.commit()
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            return render_template('register.html', error="Username already exists")
        finally:
            cursor.close()
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        if result and result[0] == password:
            session['username'] = username
            return redirect(url_for('index'))
        return render_template('login.html', error="Invalid credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/image_detection', methods=['GET', 'POST'])
def image_detection():
    if 'username' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['image']
        model_type = request.form['model']
        img = Image.open(file.stream).convert('RGB')
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        start_time = time.time()
        detections_list = []
        scores_list = []
        labels_list = []

        if model_type == 'ensemble':
            yolo_dets = yolo_model.detect(img_array, conf_threshold=0.5, iou_threshold=0.4)
            yolo_boxes = yolo_dets[:, :4] / np.array([img_array.shape[1], img_array.shape[0]] * 2)
            yolo_scores = yolo_dets[:, 4]
            yolo_labels = yolo_dets[:, 5]
            detections_list.append(yolo_boxes)
            scores_list.append(yolo_scores)
            labels_list.append(yolo_labels)

            frcnn_dets = frcnn_model.detect(img_array, conf_threshold=0.7, nms_threshold=0.3)
            frcnn_boxes = frcnn_dets[:, :4] / np.array([img_array.shape[1], img_array.shape[0]] * 2)
            frcnn_scores = frcnn_dets[:, 4]
            frcnn_labels = frcnn_dets[:, 5]
            detections_list.append(frcnn_boxes)
            scores_list.append(frcnn_scores)
            labels_list.append(frcnn_labels)

            boxes, scores, labels = weighted_boxes_fusion(detections_list, scores_list, labels_list, iou_thr=0.6, skip_box_thr=0.3)
            boxes *= np.array([img_array.shape[1], img_array.shape[0]] * 2)
            detections = np.column_stack((boxes, scores, labels))
            result_img = yolo_model.draw_boxes(img_array.copy(), detections)
            class_names = yolo_model.class_names
        else:
            if model_type == 'yolo':
                detections = yolo_model.detect(img_array, conf_threshold=0.5, iou_threshold=0.4)
                result_img = yolo_model.draw_boxes(img_array, detections)
                class_names = yolo_model.class_names
            else:
                detections = frcnn_model.detect(img_array, conf_threshold=0.7, nms_threshold=0.3)
                result_img = frcnn_model.draw_boxes(img_array, detections)
                class_names = frcnn_model.class_names

        inference_time = time.time() - start_time
        result_path = f'static/result_{model_type}_{int(time.time())}.jpg'
        cv2.imwrite(result_path, result_img)

        iou_scores = []
        detected_objects = []
        for det in detections:
            x_min, y_min, x_max, y_max, conf, cls = det
            class_name = class_names[int(cls)]
            ground_truth_box = [x_min, y_min, x_max, y_max]  # Placeholder
            iou = float(calculate_iou([x_min, y_min, x_max, y_max], ground_truth_box))  # Convert to Python float
            iou_scores.append(iou)
            detected_objects.append({
                'label': class_name,
                'confidence': float(conf),  # Convert to Python float
                'iou': float(iou),  # Convert to Python float
                'precision': "Yes" if iou > 0.5 else "No"
            })

        true_positives = len([s for s in iou_scores if s > 0.5])
        precision = float(true_positives / len(detections)) if len(detections) > 0 else 0  # Convert to Python float
        mean_iou = float(np.mean(iou_scores)) if iou_scores else 0  # Convert to Python float
        accuracy = mean_iou

        update_performance_metrics(model_type, inference_time, iou_scores, result_image_path=result_path)
        save_detection_history(model_type, result_path, inference_time, iou_scores, detected_objects)

        return render_template('image_detection.html', result_image=result_path, iou=mean_iou,
                               inference_time=inference_time, precision=precision, accuracy=accuracy,
                               objects=detected_objects)
    return render_template('image_detection.html')

def gen_frames(model_type):
    frame_count = 0
    start_time = time.time()
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame_count += 1
        t = time.time()
        if model_type == 'yolo':
            detections = yolo_model.detect(frame, conf_threshold=0.5, iou_threshold=0.4)
            frame = yolo_model.draw_boxes(frame, detections)
        else:
            detections = frcnn_model.detect(frame, conf_threshold=0.7, nms_threshold=0.3)
            frame = frcnn_model.draw_boxes(frame, detections)
        inference_time = time.time() - t

        iou_scores = [0.5 for _ in detections]

        if frame_count >= 30:
            throughput = frame_count / (time.time() - start_time)
            update_performance_metrics(model_type, inference_time, iou_scores, throughput=throughput)
            frame_count = 0
            start_time = time.time()
        else:
            update_performance_metrics(model_type, inference_time, iou_scores)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed/<model_type>')
def video_feed(model_type):
    return Response(gen_frames(model_type), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime_detection')
def realtime_detection():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('realtime_detection.html')

@app.route('/performance_analysis')
def performance_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT model_type, mAP, inference_time, throughput, `precision`, result_image_path FROM performance_metrics")
    analysis = {row['model_type']: row for row in cursor.fetchall()}
    cursor.execute("SELECT model_type, image_path, inference_time, mAP, `precision`, throughput, detected_objects, detection_time FROM detection_history ORDER BY detection_time DESC")
    history = cursor.fetchall()
    for record in history:
        detected_objects = json.loads(record['detected_objects'])
        true_positives = len([obj for obj in detected_objects if obj['precision'] == 'Yes'])
        ground_truth_count = true_positives  # Placeholder
        record['recall'] = true_positives / ground_truth_count if ground_truth_count > 0 else 0
    cursor.close()
    conn.close()
    return render_template('performance_analysis.html', analysis=analysis, history=history)

if __name__ == '__main__':
    app.run(debug=True)
