from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from spotify_helper import get_spotify_recommendations
from PIL import Image
import io, torch

# ─────────────────────────────────────────────
# LOAD MODELS ONCE
# ─────────────────────────────────────────────
_device = "cuda" if torch.cuda.is_available() else "cpu"

_caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(_device)
_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
MOOD_LABELS = [
    "focus", "workout", "morning", "wind-down", "hype", "drive", "rainy day",
    "relaxed", "romantic", "creative flow", "melancholy", "inspired",
    "confident", "social", "sleep"
]
_mood_embeddings = _embedder.encode(MOOD_LABELS, convert_to_tensor=True)

# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────
def caption_image(image_bytes: bytes) -> str:
    """Generate a caption from the image."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = _processor(images=img, return_tensors="pt").to(_device)
    out_ids = _caption_model.generate(**inputs, max_length=30)
    return _tokenizer.decode(out_ids[0], skip_special_tokens=True)

def mood_from_caption(caption: str) -> str:
    """Find the closest mood semantically to the caption."""
    cap_emb = _embedder.encode(caption, convert_to_tensor=True)
    sims = util.cos_sim(cap_emb, _mood_embeddings)[0]
    best_idx = int(torch.argmax(sims))
    return MOOD_LABELS[best_idx]

def infer_and_recommend(image_bytes: bytes, hint: str, focus_pref: str):
    """Main logic: caption → mood → Spotify feature mapping."""
    if hint == "Auto (detect)":
        caption = caption_image(image_bytes)
        mood = mood_from_caption(caption)
        explain = f"caption='{caption}'"
    else:
        mood = hint.lower().replace(" ", "")
        explain = f"hint='{hint}'"

    presets = {
        "focus":     (["study","electronic"], 95, 0.45, 0.4),         # lofi → study
        "workout":   (["edm","hip-hop"],    135, 0.85, 0.6),
        "morning":   (["acoustic","indie"],  95, 0.35, 0.7),
        "wind-down": (["ambient","chill"],   75, 0.25, 0.4),
        "hype":      (["dance","pop"],      125, 0.8, 0.8),
        "drive":     (["synth-pop","alternative"], 105, 0.55, 0.55),  # synthwave → synth-pop
        "rainy day": (["jazz","ambient"],    80, 0.3, 0.4),
        "relaxed":   (["chill","acoustic"],  85, 0.3, 0.6),
        "romantic":  (["r-n-b","soul"],      85, 0.4, 0.6),           # rnb → r-n-b
        "creative flow": (["jazz","study"],  90, 0.45, 0.5),          # lo-fi → study
        "melancholy":(["indie","piano"],     75, 0.2, 0.3),
        "inspired":  (["pop","movies"],     110, 0.7, 0.7),           # cinematic → movies
        "confident": (["hip-hop","pop"],    120, 0.8, 0.7),
        "social":    (["dance","pop"],      125, 0.8, 0.8),
        "sleep":     (["ambient","classical"], 70, 0.1, 0.3),
        "ambiguous": (["chill"],             90, 0.4, 0.5)
    }

    sg, tt, e, v = presets.get(mood, presets["ambiguous"])
    features = {"target_tempo": tt, "target_energy": e, "target_valence": v}

    
    try: 
        res = get_spotify_recommendations(
            sg, features, limit=10, market="US", mood_for_fallback=mood
        )
        tracks, source = res["tracks"], res["source"]

    except Exception as e:
        print("Spotify error:", e)
        tracks, source = [], "error"

    return {
        "mood": mood,
        "seed_genres": sg,
        "features": features,
        "tracks": tracks,
        "source": source,   # 👈 include this key
        "explain": explain,
    }
