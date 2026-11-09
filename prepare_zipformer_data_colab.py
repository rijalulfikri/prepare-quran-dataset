"""Prepare Zipformer2-CTC training data on Colab: HF -> FLAC + Lhotse manifests on Drive.

Idempotent: re-run after a dead session and finished configs are skipped.

    python prepare_zipformer_data_colab.py --drive-root /content/drive/MyDrive/muaalem-zipformer-data
    python prepare_zipformer_data_colab.py --drive-root ... --only moshaf_6.0 --only noise
"""

import argparse
import json
import logging
import os

from prepare_quran_dataset.zipformer import data_prep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/train/zipformer/train_config.yml")
    parser.add_argument("--drive-root", required=True, help="output folder (on mounted Drive)")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        help="config to process (`moshaf_<id>` or `noise`); repeatable; default all",
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--skip-space-check", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = data_prep.run(
        args.config,
        args.drive_root,
        args.hf_token,
        only=args.only,
        skip_space_check=args.skip_space_check,
    )
    if summary is not None:
        print(json.dumps({k: summary[k] for k in ("hours", "num_cuts")}, indent=2))


if __name__ == "__main__":
    main()
