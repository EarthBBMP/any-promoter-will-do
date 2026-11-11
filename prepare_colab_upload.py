"""Bundle everything Colab needs into a single zip.

Produces ``aspine_colab.zip`` mirroring the local repository layout:

    xpresso_module/           the released model definition
    data/Aspine/*.csv         train / valid / test splits
    data/Saved_Weights/       the released weights
    aspine-newinml/           this package (code only, no cached predictions)

Because the layout matches, ``aspine_bench.config`` resolves its paths on Colab
exactly as it does locally and nothing needs patching.

The CSVs are ~296 MB of text and deflate about 3.5x, so the archive lands near
85 MB - a minute or two to upload on a normal connection.

    python prepare_colab_upload.py
"""

import zipfile
from pathlib import Path

from aspine_bench import config

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "aspine_colab.zip"

# (source path, path inside the archive)
CODE_FILES = [
    "aspine_bench/__init__.py",
    "aspine_bench/config.py",
    "aspine_bench/data.py",
    "aspine_bench/metrics.py",
    "aspine_bench/baselines.py",
    "aspine_bench/cnn.py",
    "run_benchmark.py",
    "train_cnn.py",
    "ablate_sequence.py",
    "requirements.txt",
    "tests/test_benchmark.py",
]


def main() -> None:
    members: list[tuple[Path, str]] = []

    for name in CODE_FILES:
        members.append((HERE / name, f"aspine-newinml/{name}"))

    for name in ["constants.py", "model.py"]:
        members.append((config.REPO_ROOT / "xpresso_module" / name, f"xpresso_module/{name}"))

    for csv in [config.TRAIN_CSV, config.VALID_CSV, config.TEST_CSV]:
        members.append((csv, f"data/Aspine/{csv.name}"))

    members.append((config.WEIGHTS_PATH, f"data/Saved_Weights/{config.WEIGHTS_PATH.name}"))

    missing = [str(src) for src, _ in members if not src.exists()]
    if missing:
        raise SystemExit("Missing required files:\n  " + "\n  ".join(missing))

    total_raw = sum(src.stat().st_size for src, _ in members)
    print(f"Packing {len(members)} files ({total_raw / 1e6:.0f} MB raw) -> {OUTPUT.name}")

    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for src, arcname in members:
            print(f"  {arcname}")
            archive.write(src, arcname)

    packed = OUTPUT.stat().st_size
    print(
        f"\nWrote {OUTPUT}"
        f"\n  {packed / 1e6:.0f} MB  ({total_raw / packed:.1f}x compression)"
        f"\n\nUpload this one file to Google Drive, then run colab_train.ipynb."
    )


if __name__ == "__main__":
    main()
