"""Classify images with the trained model and report confidence scores.

    python -m classifier.predict path/to/image.jpg [more images...] --top 3
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .data import eval_transform, pick_device
from .model import build_model, predict_proba

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "eurosat_mobilenet.pt"

# Below this top-1 probability the prediction is flagged as uncertain. Useful for
# images unlike the training data (e.g. a photo of a cat given to a satellite model).
LOW_CONFIDENCE = 0.5


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]

    @property
    def uncertain(self) -> bool:
        return self.confidence < LOW_CONFIDENCE


class Classifier:
    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, device=None):
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"{checkpoint} not found. Run `python -m classifier.train` first.")
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.classes = bundle["classes"]
        self.device = device or pick_device()
        self.model = build_model(len(self.classes), bundle["arch"], pretrained=False)
        self.model.load_state_dict(bundle["state_dict"])
        self.model.to(self.device).eval()
        self.transform = eval_transform(bundle["size"])

    @classmethod
    def from_model(cls, model, classes, size=128):
        """Wrap an in-memory model (used by tests)."""
        self = cls.__new__(cls)
        self.model, self.classes, self.device = model.eval(), list(classes), torch.device("cpu")
        self.transform = eval_transform(size)
        return self

    def predict(self, image: Image.Image, top: int = 3) -> Prediction:
        batch = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = predict_proba(self.model, batch)[0].cpu()
        values, indices = probs.topk(min(top, len(self.classes)))
        top_k = [(self.classes[i], float(v)) for v, i in zip(values, indices)]
        return Prediction(top_k[0][0], top_k[0][1], top_k)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args(argv)
    clf = Classifier(args.checkpoint)
    for path in args.images:
        pred = clf.predict(Image.open(path), args.top)
        flag = "  (low confidence)" if pred.uncertain else ""
        print(f"{path.name}: {pred.label} {pred.confidence:.1%}{flag}")
        for label, p in pred.top_k[1:]:
            print(f"    {label:22s} {p:.1%}")


if __name__ == "__main__":
    main()
