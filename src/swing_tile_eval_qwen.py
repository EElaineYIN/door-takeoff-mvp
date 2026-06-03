"""Run the V3 tiled+few-shot swing-door eval with Qwen3.5-plus added as a 4th VLM.

This module does NOT modify any existing source. It imports the V3 eval pipeline
unchanged and registers ``qwen`` as a 4th entry in ``MODELS_V3`` at runtime, then
delegates to ``swing_tile_eval_v3.main()``.

Qwen 3.5-plus is hosted on the Stanford inference proxy (same base URL as the
other three frontier VLMs) — only the model_id and env-var differ.

Usage::

    # qwen only
    .venv/bin/python -m src.swing_tile_eval_qwen --split val

    # all four (qwen + the original three)
    .venv/bin/python -m src.swing_tile_eval_qwen --split val \\
        --models anthropic openai gemini qwen
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from src.model_comparison import ModelSpec
from src.swing_v3 import MODELS_V3, _call_few_shot_proxy
from src.swing_tile_eval_v3 import main as tile_eval_main


QWEN_KEY = "qwen"
QWEN_MODEL_ID = "qwen3.5-plus"
QWEN_ENV_VAR = "QWEN_API_KEY"


def _register_qwen() -> None:
    """Add a 'qwen' entry to MODELS_V3 using the same few-shot proxy call."""
    if QWEN_KEY in MODELS_V3:
        return
    MODELS_V3[QWEN_KEY] = ModelSpec(
        key=QWEN_KEY,
        label="Alibaba Qwen 3.5 Plus",
        model_id=QWEN_MODEL_ID,
        env_var=QWEN_ENV_VAR,
        call=_call_few_shot_proxy,
    )


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run swing-door V3 eval (tile + upscale + few-shot) with Qwen3.5-plus added."
    )
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--upscale", type=float, default=2.0)
    p.add_argument("--postprocess", action="store_true",
                   help="Apply L->square content shrinker after NMS.")
    p.add_argument("--models", nargs="+", default=[QWEN_KEY],
                   help="Which models to run. Default: qwen only.")
    return p.parse_args()


def main() -> None:
    load_dotenv(override=True)
    _register_qwen()

    if not os.environ.get(QWEN_ENV_VAR):
        print(f"ERROR: {QWEN_ENV_VAR} is not set in environment / .env",
              file=sys.stderr)
        sys.exit(1)

    a = _args()
    tile_eval_main(split=a.split, upscale=a.upscale,
                   postprocess=a.postprocess, models=a.models)


if __name__ == "__main__":
    main()
