"""Quick diagnostic for the Gemini API call.

Run from the project root with::

    python check_gemini.py

Prints the exact error so we can tell whether the issue is the key, the
model id, the quota, or the request shape.
"""

from __future__ import annotations

import io
import sys

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Insert project root into path before importing src.*
sys.path.insert(0, ".")

from src import model_comparison


def main() -> None:
    img = Image.new("RGB", (500, 500), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    run = model_comparison.run_plan_tile_model(
        "gemini", buf.getvalue(), use_cache=False
    )

    print("=" * 60)
    print("Gemini diagnostic")
    print("=" * 60)
    print(f"error:    {run.error or '(none)'}")
    print(f"cached:   {run.cached}")
    print(f"elapsed:  {run.elapsed_s:.2f}s")
    print(f"df rows:  {len(run.df)}")
    raw = run.raw_response or ""
    print(f"raw len:  {len(raw)}")
    print("raw[:600]:")
    print(raw[:600] if raw else "(empty)")


if __name__ == "__main__":
    main()
