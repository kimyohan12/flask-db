from models.yolo import attempt_load
from flask import Flask, render_template, request, make_response, url_for
import os
from werkzeug.utils import secure_filename
import torch
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.plots import plot_one_box
from PIL import Image
import numpy as np
import base64
import io
import cv2
from torchvision import transforms


app = Flask(__name__, static_folder='static/images', static_url_path='/static')
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# 객체 감지를 위한 변수들
img_size = 640  # 이미지 크기
conf_thres = 0.85  # confidence threshold 값
iou_thres = 0.45  # IoU threshold 값
agnostic = False  # agnostic NMS를 사용할 지 여부
augment = False  # augmentation을 사용할 지 여부
half = False  # half precision을 사용할 지 여부
names = None

# Set model to evaluation mode
weights = 'models/best.pt'  # 모델 가중치 파일 경로
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, device)  # 모델 가중치 파일 로딩
model.to(device)
model.eval() 


# Class 색상 리스트
colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 255, 255]]

# 웹캠에서 이미지 가져오기
cap = cv2.VideoCapture(0)

# 이미지 전처리 함수
def preprocess(img):
    pass


@app.route('/')
def home():
    input_image_path = url_for('static', filename='input_image.jpg')
    output_image_path = url_for('static', filename='output_image.jpg')
    return render_template('home.html', input_image_path=input_image_path, output_image_path=output_image_path)


@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'images' not in request.files:
        return 'No file uploaded', 400
    # 이미지를 POST로 받아와서 PIL Image 객체로 변환
    img = Image.open(request.files['images'].stream).convert('RGB')
    
    # 이미지를 전처리
    img = preprocess(img)

    # 이미지 크기를 체크하고 조절
    img, ratio, pad = check_img_size(img, img_size=img_size, stride=32)
    
    # 이미지를 tensor로 변환
    img = transforms.ToTensor()(img)
    
    # 이미지를 모델에 전달하여 객체 감지 수행
    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=agnostic)
        
    # 감지된 객체들을 이미지 위에 시각화
    if len(pred[0]):
        pred_class = pred[0][:, :-1]
        pred_class[:, :4] = scale_coords(img.shape[1:], pred_class[:, :4], img.shape[2:]).round()
        for c in pred[0][:, -1].unique():
            n = (pred[0][:, -1] == c).sum()
            text = f"{names[int(c)]} {n}"
            pred_c = pred_class[pred[0][:, -1] == c]
            for *xyxy, conf in pred_c:
                label = f"{text} {conf:.2f}"
                plot_one_box(xyxy, img.clone(), label=label, color=colors[int(c)], line_thickness=3)
    
        # 이미지를 PIL Image로 변환
        img = transforms.ToPILImage()(img.cpu())
    
        # 변환된 이미지를 response로 반환
        response = make_response()
        response.data = io.BytesIO()
        img.save(response.data, "JPEG")
        response.headers['Content-Type'] = 'image/jpeg'
    
        return response
# Get detections
    with torch.no_grad():
        detections = model(img, augment=augment)[0]

        detections = non_max_suppression(detections, conf_thres=conf_thres, iou_thres=iou_thres,  agnostic=agnostic)

        # 감지 결과 후처리
        detections = non_max_suppression(detections, conf_thres, iou_thres, agnostic)

        # bounding box 그리기
        for det in detections:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img.shape[2:]).round()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls) # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, img, label=label, color=colors(c, True), line_thickness=3)

    # 이미지 저장
    result_path = os.path.join(app.static_folder, 'result.jpg')
    img = img.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]  # to numpy
    Image.fromarray(img).save(result_path)

    # HTML 페이지에서 감지 렌더링
    img_base64 = base64.b64encode(open(result_path, 'rb').read()).decode('ascii')
    return render_template('detections.html', img_base64=img_base64, detections=detections)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    
    app.run(host='192.168.1.185', port=9795, debug=True)