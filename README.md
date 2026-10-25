` AI Image Generation Studio

**DecodeLabs Generative AI Internship — Project 3**

A text-to-image studio with a **web interface (Streamlit)** and a **command-line tool**. Describe an image, choose the aspect ratio, style and number of images, and the studio generates, downloads, verifies and saves them.

It works out of the box with the free **Pollinations (FLUX)** API — no key needed — and also supports **Stability AI (Stable Image Core)**.

`` Features
- **Aspect ratio → exact pixels** — user choices are mapped to sizes the models accept (`16:9 → 1344×768`, `1:1 → 1024×1024`, `9:16 → 768×1344`…)
- **Style presets** — photorealistic, digital art, cyberpunk, minimalist, watercolor, anime
- **Multiple images** (1–4) with different seeds, or a fixed seed for repeatable results
- **Negative prompt** support
- **Split timeouts** — `(3.05s connect, 120s read)`: fail fast if the server is down, wait patiently while the GPU generates
- **Retries with exponential backoff + jitter** on busy servers (429/5xx), read timeouts and corrupt downloads
- **Memory-safe streaming** — images are written to disk in 64 KB chunks, never loaded fully into RAM
- **Integrity check** — `Image.open().load()` decodes every pixel, so a download cut off halfway is detected (header-only checks like `verify()` miss it)
- **Safety filter handling** — blocked prompts show a friendly message instead of crashing, and are not retried
- **Metadata** — every run saves a `metadata.json` with the prompt, settings, seeds and timings
- **Rate-limit aware** — the free tier allows about one image every 45 seconds, so the studio waits between images automatically; when the service is full it returns `402` with a `Retry-After` header, which the studio respects before retrying
- **Response check** — only `image/*` responses are accepted; an error page or JSON is retried instead of being saved as a broken file

> **Note on the free tier:** Pollinations may return images smaller than requested (e.g. 1015×580 for a 1344×768 request) while keeping the aspect ratio. The real size is checked and saved in `metadata.json`. Stability AI returns full resolution.

`` How It Works
```
INPUT                          PROCESS                               OUTPUT
prompt + style       →   request with split timeouts      →   stream to disk (64 KB chunks)
aspect ratio → pixels    retry with backoff + jitter           full pixel decode (.load())
count / seed             safety-block detection                save image + metadata.json
```

`` Setup
Requires Python 3.10+.
```bash
git clone https://github.com/Moemenakari/image-studio.git
cd image-studio
pip install -r requirements.txt
```
No API key is needed for the default provider. To use Stability AI instead, copy `.env.example` to `.env` and set:
```
IMAGE_PROVIDER=stability
STABILITY_API_KEY=your_key_here
```

`` Usage
**Web app:**
```bash
streamlit run app.py
```

**Command line:**
```bash
python generate.py "a cedar tree on a snowy mountain at sunset" --ratio 16:9 --style digital-art
python generate.py "a cozy coffee shop on a rainy day" --count 3 --style watercolor --seed 42
python generate.py "a futuristic Beirut skyline" --style cyberpunk --negative "blurry, text"
```
Images are saved in `outputs/<timestamp>/`.

`` Tests
The tests use a fake network session, so they run offline:
```bash
python -m unittest test_studio.py
```
They cover: aspect-ratio mapping, input validation, multiple seeds, split timeouts and streaming, retrying a busy server (429/5xx) or a full free tier (402 with `Retry-After`), rejecting non-image responses, detecting and retrying a truncated image, not retrying safety blocks, and failing fast on connection timeouts.

`` Project Structure
```
app.py               Streamlit web interface
generate.py          command-line interface
studio/config.py     aspect ratios, style presets, limits, timeouts
studio/pipeline.py   request building, providers, streaming, verification, retries
test_studio.py       unit tests
```

`` Tech Stack
Python · Streamlit · Requests · Pillow · Pollinations (FLUX) / Stability AI

`` Author
**Mouemen Akkarie** — [GitHub](https://github.com/Moemenakari)
