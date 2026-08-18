"""Lightweight tests for dataset loading. Does not require TensorFlow or model weights."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import load_dataset


def test_load_dataset_finds_all_classes():
    image_paths, labels = load_dataset()
    assert set(labels) == {"cats", "dogs", "panda"}


def test_load_dataset_pairs_match_in_length():
    image_paths, labels = load_dataset()
    assert len(image_paths) == len(labels)
    assert len(image_paths) > 0


def test_load_dataset_only_returns_image_files():
    image_paths, _ = load_dataset()
    assert all(p.lower().endswith((".jpg", ".jpeg", ".png")) for p in image_paths)
