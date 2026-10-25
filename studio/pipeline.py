"""
The generation pipeline:

  1. Build the payload   - validate the prompt, apply a style, map aspect ratio to pixels
  2. Send the request    - split timeouts + retries with exponential backoff and jitter
  3. Stream to disk      - write the image in 64 KB chunks (never the whole file in RAM)
  4. Verify the file     - force a full pixel decode so a cut-off download is caught
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image

from .config import (ASPECT_RATIOS, CHUNK_SIZE, MAX_IMAGES, MAX_PROMPT_CHARS,
                     OUTPUT_DIR, STYLE_PRESETS, TIMEOUT)

# 402 = Pollinations free tier is full right now, 429 = too many requests
RETRY_STATUS = {402, 429, 500, 502, 503, 504}


class StudioError(Exception):
    """Base error with a message that is safe to show to the user."""


class ContentBlockedError(StudioError):
    """The provider's safety filter rejected the prompt or the image."""


class CorruptImageError(StudioError):
    """The downloaded file is not a complete, valid image."""


@dataclass
class ImageRequest:
    prompt: str
    aspect_ratio: str = "1:1"
    style: str = "none"
    count: int = 1
    negative_prompt: str = ""
    seed: int | None = None
    width: int = field(init=False, default=0)
    height: int = field(init=False, default=0)
    full_prompt: str = field(init=False, default="")


def build_request(prompt, aspect_ratio="1:1", style="none", count=1, negative_prompt="", seed=None):
    """Validate the user's input and turn it into an exact request."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise StudioError("Please describe the image you want.")
    if aspect_ratio not in ASPECT_RATIOS:
        raise StudioError(f"Aspect ratio must be one of: {', '.join(ASPECT_RATIOS)}")
    if style not in STYLE_PRESETS:
        raise StudioError(f"Style must be one of: {', '.join(STYLE_PRESETS)}")
    if not 1 <= count <= MAX_IMAGES:
        raise StudioError(f"You can generate between 1 and {MAX_IMAGES} images.")

    full_prompt = f"{prompt}, {STYLE_PRESETS[style]}" if STYLE_PRESETS[style] else prompt
    if len(full_prompt) > MAX_PROMPT_CHARS:
        raise StudioError(f"Prompt is too long ({len(full_prompt)}/{MAX_PROMPT_CHARS} characters).")

    req = ImageRequest(prompt, aspect_ratio, style, count, negative_prompt.strip(), seed)
    req.width, req.height = ASPECT_RATIOS[aspect_ratio]
    req.full_prompt = full_prompt
    return req


# ---------------- Providers ----------------

class PollinationsProvider:
    """Free text-to-image API (FLUX model). No API key needed."""

    name = "pollinations"
    base_url = "https://image.pollinations.ai/prompt/"
    # The free tier allows about 1 image every 45 seconds. Going faster gets
    # requests queued or rejected (HTTP 402), so we wait between images.
    min_interval = 46

    def open_stream(self, session, req, seed):
        params = {"width": req.width, "height": req.height, "seed": seed,
                  "model": "flux", "nologo": "true"}
        if req.negative_prompt:
            params["negative_prompt"] = req.negative_prompt
        url = self.base_url + quote(req.full_prompt)
        return session.get(url, params=params, timeout=TIMEOUT, stream=True)


class StabilityProvider:
    """Stability AI - Stable Image Core (needs STABILITY_API_KEY)."""

    name = "stability"
    url = "https://api.stability.ai/v2beta/stable-image/generate/core"
    min_interval = 0

    def __init__(self, api_key):
        self.api_key = api_key

    def open_stream(self, session, req, seed):
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "image/*"}
        data = {"prompt": req.full_prompt, "aspect_ratio": req.aspect_ratio,
                "output_format": "png", "seed": seed}
        if req.negative_prompt:
            data["negative_prompt"] = req.negative_prompt
        return session.post(self.url, headers=headers, files={"none": ""}, data=data,
                            timeout=TIMEOUT, stream=True)


def get_provider(name=None):
    name = (name or os.getenv("IMAGE_PROVIDER", "pollinations")).lower()
    if name == "stability":
        key = os.getenv("STABILITY_API_KEY")
        if not key:
            raise StudioError("STABILITY_API_KEY is missing in your .env file.")
        return StabilityProvider(key)
    return PollinationsProvider()


# ---------------- Network + download ----------------

def check_response(response):
    """Turn HTTP errors into clear messages. Safety blocks are not retried."""
    if response.status_code == 200:
        content_type = response.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            # Got a JSON/HTML error page instead of an image: treat it like a busy server
            raise requests.HTTPError(f"Expected an image but got '{content_type}'", response=response)
        return
    try:
        text = response.text[:300].lower()
    except Exception:
        text = ""
    if response.status_code in (400, 403) and any(w in text for w in ("moderat", "safety", "policy", "nsfw", "content")):
        raise ContentBlockedError("This prompt was blocked by the safety filter. Try different wording.")
    if response.status_code in RETRY_STATUS:
        raise requests.HTTPError(f"Server busy ({response.status_code})", response=response)
    raise StudioError(f"The image API returned an error ({response.status_code}).")


def stream_to_file(response, path):
    """Write the image to disk chunk by chunk to keep memory usage low."""
    size = 0
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                size += len(chunk)
    return size


def verify_image(path):
    """
    Image.verify() only checks the file header, so a download that was cut off
    halfway still passes. .load() decodes every pixel, which catches it.
    """
    try:
        with Image.open(path) as img:
            img.load()
            return img.format, img.size
    except (OSError, SyntaxError) as error:
        raise CorruptImageError(f"Downloaded image is incomplete or corrupted: {error}") from error


def download_with_retry(session, provider, req, seed, path, max_attempts=4, base_delay=2.0):
    """Retry on timeouts, busy servers and corrupt files, with exponential backoff + jitter."""
    for attempt in range(1, max_attempts + 1):
        try:
            with provider.open_stream(session, req, seed) as response:
                check_response(response)
                stream_to_file(response, path)
            return verify_image(path)
        except requests.exceptions.ConnectTimeout as error:
            # Can't even connect: the server is down or unreachable. Fail fast.
            raise StudioError("Could not connect to the image server. Check your internet connection.") from error
        except ContentBlockedError:
            raise
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError,
                requests.HTTPError, CorruptImageError) as error:
            if path.exists():
                path.unlink()
            if attempt == max_attempts:
                raise StudioError(f"Generation failed after {max_attempts} attempts: {error}") from error
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
            # If the server tells us how long to wait, respect it
            response = getattr(error, "response", None)
            retry_after = response.headers.get("retry-after") if response is not None else None
            if retry_after and retry_after.isdigit():
                delay = max(delay, int(retry_after))
            time.sleep(delay)


# ---------------- Main entry point ----------------

def generate(req, provider=None, session=None, output_dir=OUTPUT_DIR, retry_delay=2.0, on_progress=None):
    """Generate req.count images and save them with a metadata file. Returns a list of results."""
    provider = provider or get_provider()
    session = session or requests.Session()
    folder = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)

    base_seed = req.seed if req.seed is not None else random.randint(1, 1_000_000)
    results = []
    for i in range(req.count):
        if i > 0 and provider.min_interval:
            if on_progress:
                on_progress(f"Waiting {provider.min_interval}s for the free tier rate limit...")
            time.sleep(provider.min_interval)
        if on_progress:
            on_progress(f"Generating image {i + 1} of {req.count}...")
        seed = base_seed + i  # a different seed gives a different variation
        temp_path = folder / f"image_{i + 1}.download"
        start = time.time()
        fmt, size = download_with_retry(session, provider, req, seed, temp_path, base_delay=retry_delay)
        # Name the file after its real format (the free API returns JPEG, Stability returns PNG)
        ext = "jpg" if fmt == "JPEG" else (fmt or "png").lower()
        path = temp_path.rename(folder / f"image_{i + 1}.{ext}")
        results.append({
            "file": str(path), "seed": seed, "format": fmt,
            "width": size[0], "height": size[1],
            "seconds": round(time.time() - start, 1),
        })

    meta = {"provider": provider.name, "request": asdict(req), "images": results,
            "created": datetime.now().isoformat(timespec="seconds")}
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return results
