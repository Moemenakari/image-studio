"""
Web interface for the studio, built with Streamlit.

Run with:  streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from studio.config import ASPECT_RATIOS, MAX_IMAGES, STYLE_PRESETS
from studio.pipeline import ContentBlockedError, StudioError, build_request, generate, get_provider

load_dotenv()

st.set_page_config(page_title="AI Image Studio", page_icon="🎨", layout="wide")
st.title("🎨 AI Image Generation Studio")
st.caption("Describe an image, pick a size and a style, and generate.")

with st.sidebar:
    st.header("Settings")
    ratio = st.selectbox("Aspect ratio", list(ASPECT_RATIOS), help="Mapped to an exact pixel size")
    w, h = ASPECT_RATIOS[ratio]
    st.caption(f"Resolution: {w} x {h}")
    style = st.selectbox("Style preset", list(STYLE_PRESETS))
    count = st.slider("Number of images", 1, MAX_IMAGES, 1)
    use_seed = st.checkbox("Fixed seed (repeatable results)")
    seed = st.number_input("Seed", min_value=1, value=42, step=1) if use_seed else None
    provider_name = st.radio("Provider", ["pollinations", "stability"],
                             help="Pollinations is free. Stability needs STABILITY_API_KEY in .env")

prompt = st.text_area("Prompt", placeholder="e.g. a cedar tree on a snowy mountain at sunset", height=100)
negative = st.text_input("Negative prompt (optional)", placeholder="e.g. blurry, text, watermark")

if st.button("Generate", type="primary", width="stretch"):
    try:
        req = build_request(prompt, ratio, style, count, negative, int(seed) if seed else None)
        status = st.empty()
        with st.spinner(f"Generating {count} image(s)..."):
            results = generate(req, get_provider(provider_name), on_progress=status.info)
        status.empty()
        st.success(f"Done! Saved to {Path(results[0]['file']).parent}")

        cols = st.columns(min(len(results), 2))
        for i, r in enumerate(results):
            with cols[i % len(cols)]:
                st.image(r["file"], caption=f"Seed {r['seed']} · {r['width']}x{r['height']} · {r['seconds']}s",
                         width="stretch")
                with open(r["file"], "rb") as f:
                    ext = Path(r["file"]).suffix
                    mime = "image/jpeg" if ext == ".jpg" else f"image/{ext[1:]}"
                    st.download_button("Download", f, file_name=f"image_{r['seed']}{ext}",
                                       mime=mime, key=f"dl{i}")
    except ContentBlockedError as error:
        st.warning(str(error))
    except StudioError as error:
        st.error(str(error))
