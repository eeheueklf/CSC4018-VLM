# detectron2
# 환경설정
!pip install 'git+https://github.com/facebookresearch/detectron2.git'

import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from google.colab.patches import cv2_imshow

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  

image_path = "/content/IMAGE/방3.jpg"  
image = cv2.imread(image_path)

predictor = DefaultPredictor(cfg)
outputs = predictor(image)

v = Visualizer(
    image[:, :, ::-1],
    metadata=None,
    scale=1.2,
    instance_mode=ColorMode.IMAGE_BW  # 배경을 흑백으로
)

out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

cv2_imshow(out.get_image()[:, :, ::-1])

instances = outputs["instances"].to("cpu")
pred_classes = instances.pred_classes
pred_scores = instances.scores

class_names = [
    "chair", "couch", "potted plant", "bed", "dining table", "tv", "lamp", 
    "armchair", "coffee table", "bookshelf", "wardrobe", "side table", "nightstand", 
    "dresser", "sofa", "ottoman", "tv stand", "cabinet", "console table", "stool", 
    "recliner", "bean bag", "rocking chair", "bench", "bar stool", 
    "storage chest", "chest of drawers", "desk", "writing desk", "curtain", "carpet",
    "table", "stairs", "plant", "pillow", "window", "frame", "refrigerator",
    "computer", "bookcase", "drawer", "mirror", "painting", "shelf", "chandelier", 
    "clock", "vase", "rug", "cushion", "sideboard", "counter", "tray", "mat", 
    "bar cart", "tv cabinet", "ottoman stool", "floor lamp", "reading chair", 
    "futon", "side chair", "end table", "wall art", "laundry basket", "shoe rack", 
    "blanket", "coffee tray", "fireplace", "staircase", "doormat", "trolley", 
    "hallway table", "storage bench", "armchair lounge", "floating shelves"
]


for class_id, score in zip(pred_classes, pred_scores):
    if class_id < len(class_names):
        print(f"Detected object: {class_names[class_id]} with score: {score:.3f}")
    else:
        print(f"Detected object: Unknown class ID {class_id} with score: {score:.3f}")



#다시시도
import numpy as np
import cv2
from sklearn.cluster import KMeans
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

# Detectron2 설정
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 신뢰도 임계값
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 모델 로드
predictor = DefaultPredictor(cfg)

# 이미지 로드
image_path = "/content/image/방3.jpg"  # 이미지 경로
image = cv2.imread(image_path)
outputs = predictor(image)

# 객체 마스크와 클래스 정보 추출
instances = outputs["instances"].to("cpu")
masks = instances.pred_masks.numpy()
classes = instances.pred_classes.numpy()

# 색상 이름 매핑용 함수
def rgb_to_color_name(rgb):
    # 간단한 색상 이름 매핑 (필요시 추가)
    color_map = {
        "흰색": [255, 255, 255],
        "검은색": [0, 0, 0],
        "파란색": [0, 0, 255],
        "빨간색": [255, 0, 0],
        "초록색": [0, 255, 0],
        "노란색": [255, 255, 0],
    }
    # 가장 가까운 색상 이름 찾기
    distances = {name: np.linalg.norm(np.array(rgb) - np.array(color)) for name, color in color_map.items()}
    return min(distances, key=distances.get)

# 객체별 대표 색상 추출
def extract_color_from_mask(image, mask):
    masked_pixels = image[mask]  # 마스크가 적용된 픽셀
    if len(masked_pixels) == 0:
        return [0, 0, 0]  # 기본 색상 (없음)
    kmeans = KMeans(n_clusters=1).fit(masked_pixels)  # K-means로 대표 색상 계산
    return kmeans.cluster_centers_[0].astype(int)

# 객체별 색상 분석
for i, mask in enumerate(masks):
    color = extract_color_from_mask(image, mask)
    color_name = rgb_to_color_name(color)
    print(f"Object {i}: Class {classes[i]}, Color {color_name} (RGB: {color})")
# 필요한 클래스만 정의
custom_classes = [
    "chair", "couch", "bed", "dining table", "tv", "lamp", "bookshelf", "potted plant"
]

# COCO 클래스 정의
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

#????????????????추가
# 클래스 이름 수동 설정
MetadataCatalog.get("custom_dataset").thing_classes = custom_classes

# 데이터셋 메타데이터 가져오기
custom_metadata = MetadataCatalog.get("custom_dataset")
#?????????????

# Detectron2의 출력에서 필요한 클래스만 필터링
outputs = predictor(image)
instances = outputs["instances"].to("cpu")
filtered_instances = [
    (coco_classes[class_id], score)
    for class_id, score in zip(instances.pred_classes.numpy(), instances.scores.numpy())
    if coco_classes[class_id] in custom_classes
]

# 필터링된 결과 출력
for class_name, score in filtered_instances:
    print(f"Detected object: {class_name} with score: {score:.3f}")
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer  # DefaultTrainer 가져오기

# 커스텀 COCO 데이터셋 등록
register_coco_instances("custom_dataset", {}, "path/to/custom_annotations.json", "path/to/images")

# 데이터셋 메타데이터 가져오기
custom_metadata = MetadataCatalog.get("custom_dataset")

# 모델 설정
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("custom_dataset",)
cfg.DATASETS.TEST = ()
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(custom_classes)  # 필요한 클래스 개수 설정
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# 모델 파인튜닝 (선택)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
