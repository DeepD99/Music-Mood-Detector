# judge_llm.py (local judge via Ollama + Qwen2.5-VL)
import base64, json, requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-vl"

SYSTEM = (
  "You're a music-mood judge. Given an image and a draft analysis "
  "(caption, initial mood, seed genres, features), output STRICT JSON:\n"
  "{ 'final_mood': <one of ['focus','workout','morning','wind-down','hype','drive',"
  "'rainy day','relaxed','romantic','creative flow','melancholy','inspired','confident','social','sleep','ambiguous']>,"
  "  'reasons': <short string>,"
  "  'seed_genres': [<1-3 lowercase spotify-like genres>],"
  "  'features': { 'target_tempo': <int>, 'target_energy': <0-1>, 'target_valence': <0-1> } }"
)

def _b64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")

def judge_image(image_bytes: bytes, draft: dict, temperature: float = 0.2) -> dict:
    """
    draft = {
      'caption': str, 'mood': str,
      'seed_genres': list[str], 'features': {'target_tempo': int, 'target_energy': float, 'target_valence': float},
      'entities': str  # optional text summary you computed
    }
    """
    img64 = _b64(image_bytes)
    user_text = (
        "Analyze the image and revise the draft if needed. "
        "Return ONLY JSON (no markdown) matching the schema. "
        f"Draft: {json.dumps(draft, ensure_ascii=False)}"
    )
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image", "image": f"data:image/jpeg;base64,{img64}"}
                ],
            },
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=body, timeout=120)
    r.raise_for_status()
    content = r.json().get("message", {}).get("content", "")
    # Model might wrap in code fences—strip and parse.
    content = content.strip().strip("`").replace("json\n", "")
    try:
        return json.loads(content)
    except Exception:
        # Fallback: return empty dict so caller can ignore
        return {}
