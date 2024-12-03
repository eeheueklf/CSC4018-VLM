# OPEN AI CLIP 사용
# 환경 설정
!pip install --upgrade torch torchvision torchaudio
!pip install --upgrade transformers

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 인테리어 스타일 키워드 리스트
style_keywords = [
    "modern", "minimalist", "scandinavian", "industrial", "vintage",
    "traditional", "rustic", "luxurious", "colorful", "neutral tone", "warm tone",
    "cool tone", "earthy tone", "pastel", "monochromatic", "bright",
    "muted colors", "grayscale", "beige and cream","natural fibers", "wood",
    "metal finishes", "glass surfaces", "velvet", "leather", "linen fabrics",
    "curved furniture", "cozy", "relaxing", "romantic", "urban", "tropical", "chic", 
    "art deco", "natural", "plant", "nature-inspired", "large mirrors", "high ceilings", 
]

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def extract_interior_style(image_path, keywords):
    image = load_image(image_path)
    
    inputs = processor(text=keywords, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image  
    probs = logits_per_image.softmax(dim=1)     
    
    keyword_probs = {keyword: prob.item() for keyword, prob in zip(keywords, probs[0])}
    
    sorted_keywords = sorted(keyword_probs.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords


image_path = "/content/IMAGE/주방1.jpg"  
top_styles = extract_interior_style(image_path, style_keywords)

print("Extracted Interior Styles:")
for style, prob in top_styles[:5]:  
    print(f"{style}: {prob:.4f}")