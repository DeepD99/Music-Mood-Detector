# captioner.py
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import io, torch

# ── Load model once ───────────────────────────────────────────
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_name = "nlpconnect/vit-gpt2-image-captioning"

model = VisionEncoderDecoderModel.from_pretrained(_model_name).to(_device)
processor = ViTImageProcessor.from_pretrained(_model_name)
tokenizer = AutoTokenizer.from_pretrained(_model_name)

# ── Generate a caption ─────────────────────────────────────────
def caption_image(image_bytes: bytes) -> str:
    """Return a natural-language caption of the photo."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(_device)
    out_ids = model.generate(**inputs, max_length=30)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)
