"""Settings for the image studio: sizes, styles, limits and timeouts."""

# Each aspect ratio maps to an exact pixel size the models accept.
# Sending a random size can make the API reject the request, so we never do that.
ASPECT_RATIOS = {
    "1:1": (1024, 1024),   # avatars, product grids
    "16:9": (1344, 768),   # banners, presentations
    "9:16": (768, 1344),   # reels, phone wallpapers
    "4:3": (1152, 896),
    "3:4": (896, 1152),
}

# Style presets are added to the end of the prompt.
STYLE_PRESETS = {
    "none": "",
    "photorealistic": "photorealistic, natural lighting, highly detailed, 35mm photo",
    "digital-art": "digital art, vibrant colors, clean lines, trending on artstation",
    "cyberpunk": "cyberpunk, neon lights, futuristic city, rain reflections, cinematic",
    "minimalist": "minimalist, flat design, simple shapes, soft pastel colors, lots of white space",
    "watercolor": "watercolor painting, soft edges, paper texture, gentle colors",
    "anime": "anime style, cel shading, expressive, studio quality",
}

MAX_PROMPT_CHARS = 1000
MAX_IMAGES = 4

# (connect timeout, read timeout) in seconds.
# Connect fails fast if the server is down; read waits for the slow GPU generation.
TIMEOUT = (3.05, 120)

CHUNK_SIZE = 65536  # 64 KB chunks when streaming the image to disk

OUTPUT_DIR = "outputs"
