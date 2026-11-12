import numpy as np
import torch
from PIL import Image

from classifier.data import eval_transform, stratified_splits, train_transform
from classifier.model import build_model, count_trainable, freeze_backbone
from classifier.predict import Classifier

CLASSES = ["a", "b", "c"]


def tiny_model():
    return build_model(len(CLASSES), "mobilenet_v3_small", pretrained=False)  # no download in tests


def test_transforms_shape_and_normalisation():
    img = Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8))
    for tf in (train_transform(96), eval_transform(96)):
        x = tf(img)
        assert x.shape == (3, 96, 96)
        assert x.abs().max() < 3  # normalised, not raw 0-255 pixels


def test_stratified_splits_are_disjoint_and_balanced():
    labels = np.repeat(np.arange(4), 100)
    tr, va, te = stratified_splits(labels, val=0.15, test=0.15)
    assert len(set(tr) | set(va) | set(te)) == 400
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert set(np.bincount(labels[te])) == {15}
    small = stratified_splits(labels, limit_per_class=20)
    assert sum(map(len, small)) == 80


def test_head_replaced_and_freezing():
    model = tiny_model()
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, len(CLASSES))
    total = count_trainable(model)
    freeze_backbone(model, trainable_blocks=0)
    head_only = count_trainable(model)
    freeze_backbone(model, trainable_blocks=2)
    partial = count_trainable(model)
    assert head_only < partial < total
    assert all(not p.requires_grad for p in model.features[0].parameters())


def test_prediction_confidences_are_a_distribution():
    clf = Classifier.from_model(tiny_model(), CLASSES, size=64)
    pred = clf.predict(Image.new("RGB", (80, 50), "green"), top=5)
    probs = [p for _, p in pred.top_k]
    assert len(pred.top_k) == len(CLASSES)          # capped at number of classes
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(probs) - 1) < 1e-5
    assert pred.label == pred.top_k[0][0]
