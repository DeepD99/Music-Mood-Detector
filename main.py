import io
import os
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────
# IMPORT YOUR SEMANTIC INFERENCE LOGIC
# (This comes from mood_infer.py)
# ─────────────────────────────────────────────
from mood_infer import infer_and_recommend

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="🎵 Music Mood AI", layout="centered")

st.title("🎵 Music Mood — AI Vibe Detector")
st.caption("Snap a photo on your phone, and AI will pick a mood + music style to match it.")

# Sidebar instructions
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("""
**How to use:**
1. Open this link on your phone.  
2. Take a picture of your current vibe (desk, gym, coffee, etc.).  
3. AI will caption the photo and infer your mood.  
4. Get song recommendations based on the vibe.  
""")

# ─────────────────────────────────────────────
# IMAGE CAPTURE
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📷 Camera", "📂 Upload"])

img_bytes = None
with tab1:
    cam = st.camera_input("Take a photo")
    if cam:
        img_bytes = cam.getvalue()

with tab2:
    up = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if up:
        img_bytes = up.read()

# ─────────────────────────────────────────────
# OPTIONAL MOOD OVERRIDE (user hint)
# ─────────────────────────────────────────────
with st.expander("Optional: guide the AI", expanded=False):
    intent_hint = st.selectbox(
        "Mood hint",
        [
            "Auto (detect)",
            "Focus", "Workout", "Morning", "Wind-down", "Hype",
            "Drive", "Relaxed", "Romantic", "Creative Flow", "Rainy Day",
            "Inspired", "Confident", "Melancholy", "Social", "Sleep"
        ],
        index=0
    )

    focus_style = st.radio(
        "Focus style (only for Focus mood)",
        ["Auto", "Chill focus", "Driving focus"],
        index=0
    )

# ─────────────────────────────────────────────
# WHEN IMAGE AVAILABLE → RUN PIPELINE
# ─────────────────────────────────────────────
if img_bytes:
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    st.image(image, caption="Your photo", use_container_width=True)

    try:
        if st.button("Analyze mood"):
            result = infer_and_recommend(img_bytes, hint=intent_hint, focus_pref=focus_style)
            mood = result["mood"]

            # get the spotify results + source label
            tracks = result["tracks"]
            source = result.get("source", "unknown")

            st.subheader(f"Mood detected: {mood}")
            st.caption(f"Source: {source}")  # 👈 show where tracks came from

            if tracks:
                for t in tracks:
                    st.markdown(f"- **{t['name']}** — {t['artist']}  [Open]({t['url']})")
            else:
                st.info("No tracks found — try a different photo or mood.")

            st.markdown("---")
            st.caption("✨ Powered by Hugging Face Vision + Sentence Transformers")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("📸 Take or upload a picture to begin.")
