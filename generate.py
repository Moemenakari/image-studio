"""
Command-line version of the studio.

Examples:
  python generate.py "a cedar tree on a mountain at sunset" --ratio 16:9 --style digital-art
  python generate.py "a cozy coffee shop" --count 3 --style watercolor --seed 42
"""

import argparse
import sys

from dotenv import load_dotenv

from studio.config import ASPECT_RATIOS, MAX_IMAGES, STYLE_PRESETS
from studio.pipeline import StudioError, build_request, generate, get_provider

load_dotenv()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Turn a text description into images.")
    parser.add_argument("prompt", help="What you want to see")
    parser.add_argument("--ratio", default="1:1", choices=list(ASPECT_RATIOS))
    parser.add_argument("--style", default="none", choices=list(STYLE_PRESETS))
    parser.add_argument("--count", type=int, default=1, help=f"1 to {MAX_IMAGES} images")
    parser.add_argument("--negative", default="", help="Things to avoid in the image")
    parser.add_argument("--seed", type=int, help="Same seed + same prompt = same image")
    parser.add_argument("--provider", choices=["pollinations", "stability"])
    args = parser.parse_args(argv)

    try:
        req = build_request(args.prompt, args.ratio, args.style, args.count, args.negative, args.seed)
        provider = get_provider(args.provider)
        print(f"Generating {req.count} image(s) at {req.width}x{req.height} with {provider.name}...")
        for r in generate(req, provider, on_progress=lambda msg: print("  " + msg)):
            print(f"  saved {r['file']}  ({r['width']}x{r['height']}, seed {r['seed']}, {r['seconds']}s)")
    except StudioError as error:
        print(f"Error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
