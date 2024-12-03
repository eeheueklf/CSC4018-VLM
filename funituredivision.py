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
