"""Save one test-set image per class to examples/ for the Streamlit demo.

Uses the same seeded split as training, so these images were never trained on.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from classifier.data import load_eurosat, stratified_splits  # noqa: E402


def main():
    dataset = load_eurosat()
    _, _, test_idx = stratified_splits(dataset.targets, seed=42)
    out = ROOT / "examples"
    out.mkdir(exist_ok=True)
    seen = set()
    for i in test_idx:
        label = dataset.classes[dataset.targets[i]]
        if label in seen:
            continue
        image, _ = dataset[i]
        image.resize((256, 256)).save(out / f"{label}.jpg", quality=92)
        seen.add(label)
        if len(seen) == len(dataset.classes):
            break
    print(f"Saved {len(seen)} examples to {out}")


if __name__ == "__main__":
    main()
