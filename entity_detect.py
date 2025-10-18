# entity_detect.py
# Layer 1: zero-shot detection (OWL-ViT) to find "flag/person/skyline/logo"
# Layer 2: CLIP text-image similarity to guess specific identity (e.g., "flag of Japan")

from PIL import Image
import io, torch
from typing import List, Dict, Any

# --- OWL-ViT (zero-shot object detection) ---
from transformers import OwlViTProcessor, OwlViTForObjectDetection
_owl_proc = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
_owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# --- CLIP (zero-shot classification) ---
from transformers import CLIPProcessor, CLIPModel
_clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
_clip_proc   = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ---- Small label banks (expand as you like) ----
COUNTRIES = [
    "United States", "Japan", "India", "United Kingdom", "France",
    "Germany", "Brazil", "Italy", "Spain", "Canada", "China", "South Korea", "Mexico"
]
FLAGS_TXT = [f"the flag of {c}" for c in COUNTRIES]

LANDMARKS = [
    "Eiffel Tower", "Statue of Liberty", "Golden Gate Bridge", "Big Ben",
    "Sydney Opera House", "Mount Fuji", "Taj Mahal", "Great Wall of China",
    "Burj Khalifa", "Christ the Redeemer", "Colosseum"
]
CITIES = [
    "New York City skyline", "Paris skyline", "London skyline", "Tokyo skyline",
    "San Francisco skyline", "Dubai skyline", "Rio de Janeiro skyline"
]
CELEBS = [
    "Taylor Swift", "Barack Obama", "Lionel Messi", "Rihanna", "Drake",
    "Beyoncé", "Elon Musk", "Kim Kardashian", "LeBron James", "Justin Bieber"
]

def _clip_best_match(image: Image.Image, labels: List[str]) -> Dict[str, Any]:
    """Return best (label, score)."""
    inputs = _clip_proc(text=labels, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = _clip_model(**inputs)
        # logits_per_image: [1, len(labels)]
        scores = out.logits_per_image.softmax(dim=1)[0]
        idx = int(scores.argmax())
        return {"label": labels[idx], "score": float(scores[idx])}

def _crop(img: Image.Image, box_xyxy):
    x0, y0, x1, y1 = [max(0, int(v)) for v in box_xyxy]
    x1 = min(img.width, x1); y1 = min(img.height, y1)
    return img.crop((x0, y0, x1, y1))

def detect_entities(image_bytes: bytes, score_thresh: float = 0.22) -> Dict[str, Any]:
    """
    Returns:
      {
        "detections": [
           {"type":"flag","score":0.71,"box":[x0,y0,x1,y1],"identity":{"label":"the flag of Japan","score":0.63}},
           {"type":"person","score":0.64,"box":[...],"identity":{"label":"Taylor Swift","score":0.41}},
           {"type":"skyline","score":0.58,"box":[...],"identity":{"label":"New York City skyline","score":0.36}}
        ]
      }
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # --- Layer 1: OWL-ViT detects candidate types ---
    queries = [["flag", "person", "skyline", "logo", "car logo", "brand logo"]]
    inputs = _owl_proc(text=queries, images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = _owl_model(**inputs)

    # Convert outputs to boxes/scores/labels
    target_sizes = torch.tensor([img.size[::-1]])  # (h, w)
    results = _owl_proc.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]

    detections = []
    for box, score, lab_idx in zip(boxes, scores, labels):
        sc = float(score)
        if sc < score_thresh:
            continue
        det_type = queries[0][int(lab_idx)]
        b = [float(x) for x in box.tolist()]
        crop = _crop(img, b)

        identity = None
        # --- Layer 2: CLIP identity guess per type ---
        try:
            if det_type == "flag":
                identity = _clip_best_match(crop, FLAGS_TXT)
            elif det_type == "skyline":
                identity = _clip_best_match(crop, CITIES + LANDMARKS)
            elif det_type == "person":
                # Very rough celebrity guess; for production, use a proper face-rec model.
                identity = _clip_best_match(crop, CELEBS)
            elif det_type in ("logo", "car logo", "brand logo"):
                # Minimal demo: guess among a few common brands (expand this list)
                BRANDS = ["Nike logo", "Apple logo", "Adidas logo", "Coca-Cola logo", "Lamborghini logo", "Ferrari logo"]
                identity = _clip_best_match(crop, BRANDS)
        except Exception:
            identity = None

        detections.append({
            "type": det_type,
            "score": sc,
            "box": b,
            "identity": identity
        })

    return {"detections": detections}
