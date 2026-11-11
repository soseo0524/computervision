"""KNN experiments on CIFAR-10.

This file intentionally implements KNN, stratified 5-fold cross-validation,
and the classification metrics from scratch (NumPy only).

Example:
    python knn.py --download --data-dir ./data
    python knn.py --data-dir ./data --k-values 1 3 5 7 \
        --metrics euclidean manhattan cosine

The defaults use a stratified subset because exact KNN on all 50,000 CIFAR-10
training images is computationally expensive.  Use ``--max-train 0`` and
``--max-test 0`` to use the complete dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass
class Result:
    stage: str
    metric: str
    k: int
    fold: int | str
    accuracy: float
    precision: float
    recall: float
    f1: float
    seconds: float


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_cifar10(data_dir: Path) -> Path:
    """Download and safely extract the official CIFAR-10 Python archive."""
    extracted = data_dir / "cifar-10-batches-py"
    if extracted.is_dir():
        return extracted

    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "cifar-10-python.tar.gz"
    if not archive.exists() or file_md5(archive) != CIFAR10_MD5:
        print(f"Downloading {CIFAR10_URL}")
        urllib.request.urlretrieve(CIFAR10_URL, archive)
    if file_md5(archive) != CIFAR10_MD5:
        raise RuntimeError("CIFAR-10 archive checksum does not match.")

    with tarfile.open(archive, "r:gz") as tar:
        root = data_dir.resolve()
        for member in tar.getmembers():
            target = (data_dir / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError("Unsafe path found in archive.")
        tar.extractall(data_dir, filter="data")
    return extracted


def _read_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as file:
        batch = pickle.load(file, encoding="bytes")
    return np.asarray(batch[b"data"], dtype=np.uint8), np.asarray(batch[b"labels"])


def load_cifar10(data_dir: Path, download: bool) -> tuple[np.ndarray, ...]:
    root = data_dir / "cifar-10-batches-py"
    if not root.is_dir():
        if not download:
            raise FileNotFoundError(
                f"Dataset not found at {root}. Run again with --download."
            )
        root = download_cifar10(data_dir)

    train_batches = [_read_batch(root / f"data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([item[0] for item in train_batches])
    y_train = np.concatenate([item[1] for item in train_batches])
    x_test, y_test = _read_batch(root / "test_batch")
    return x_train, y_train, x_test, y_test


def stratified_subset(
    x: np.ndarray, y: np.ndarray, limit: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or limit >= len(y):
        return x, y
    indices: list[np.ndarray] = []
    classes = np.unique(y)
    base, remainder = divmod(limit, len(classes))
    for position, label in enumerate(classes):
        candidates = np.flatnonzero(y == label)
        take = base + (position < remainder)
        indices.append(rng.choice(candidates, size=take, replace=False))
    selected = np.concatenate(indices)
    rng.shuffle(selected)
    return x[selected], y[selected]


def make_stratified_folds(
    y: np.ndarray, n_splits: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """Return validation indices with each class distributed across folds."""
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(n_splits)]
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for fold, part in enumerate(np.array_split(indices, n_splits)):
            fold_parts[fold].append(part)
    folds = [np.concatenate(parts) for parts in fold_parts]
    for fold in folds:
        rng.shuffle(fold)
    return folds


def preprocess(images: np.ndarray, image_size: int) -> np.ndarray:
    """Convert CIFAR's CHW vectors to normalized, optionally downsampled vectors."""
    images = images.reshape(-1, 3, 32, 32)
    if image_size < 32:
        if 32 % image_size:
            raise ValueError("--image-size must divide 32 (e.g. 8, 16, or 32).")
        block = 32 // image_size
        images = images.reshape(-1, 3, image_size, block, image_size, block).mean(
            axis=(3, 5), dtype=np.float32
        )
    return np.ascontiguousarray(images.reshape(len(images), -1), dtype=np.float32) / 255.0


class KNearestNeighbors:
    """A from-scratch KNN classifier with batched exact distance search."""

    SUPPORTED_METRICS = {"euclidean", "manhattan", "chebyshev", "cosine"}

    def __init__(self, metric: str = "euclidean", batch_size: int = 128):
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"Unknown metric: {metric}")
        self.metric = metric
        self.batch_size = batch_size

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        self.x_train = np.asarray(x, dtype=np.float32)
        self.y_train = np.asarray(y, dtype=np.int64)
        if self.metric == "euclidean":
            self.train_aux = np.sum(self.x_train**2, axis=1)
        elif self.metric == "cosine":
            norms = np.linalg.norm(self.x_train, axis=1, keepdims=True)
            self.x_train = self.x_train / np.maximum(norms, 1e-12)
            self.train_aux = None
        else:
            self.train_aux = None
        return self

    def _distances(self, query: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            # Squared Euclidean distance preserves neighbor ordering and avoids sqrt.
            distances = (
                np.sum(query**2, axis=1, keepdims=True)
                + self.train_aux[None, :]
                - 2.0 * query @ self.x_train.T
            )
            return np.maximum(distances, 0.0)
        if self.metric == "cosine":
            query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
            return 1.0 - query @ self.x_train.T

        # Manhattan (L1) and Chebyshev (L-infinity) need the full difference
        # tensor, so keep its temporary array bounded with a training-data chunk.
        reduce = np.sum if self.metric == "manhattan" else np.max
        distances = np.empty((len(query), len(self.x_train)), dtype=np.float32)
        train_chunk = max(1, min(512, 2_000_000 // max(1, query.shape[1])))
        for start in range(0, len(self.x_train), train_chunk):
            stop = min(start + train_chunk, len(self.x_train))
            distances[:, start:stop] = reduce(
                np.abs(query[:, None, :] - self.x_train[None, start:stop, :]), axis=2
            )
        return distances

    def predict_many_k(self, x: np.ndarray, k_values: Iterable[int]) -> dict[int, np.ndarray]:
        k_values = sorted(set(int(k) for k in k_values))
        if not k_values or k_values[0] < 1 or k_values[-1] > len(self.x_train):
            raise ValueError("Every k must be between 1 and the training-set size.")
        output = {k: np.empty(len(x), dtype=np.int64) for k in k_values}
        max_k = max(k_values)
        n_classes = int(self.y_train.max()) + 1

        for start in range(0, len(x), self.batch_size):
            stop = min(start + self.batch_size, len(x))
            distances = self._distances(np.asarray(x[start:stop], dtype=np.float32))
            neighbor_idx = np.argpartition(distances, max_k - 1, axis=1)[:, :max_k]
            # Sort the selected neighbors so predictions for every smaller k are valid.
            selected_dist = np.take_along_axis(distances, neighbor_idx, axis=1)
            order = np.argsort(selected_dist, axis=1)
            neighbor_labels = self.y_train[np.take_along_axis(neighbor_idx, order, axis=1)]

            for k in k_values:
                for row, labels in enumerate(neighbor_labels[:, :k]):
                    # np.argmax gives deterministic smallest-label tie breaking.
                    output[k][start + row] = np.bincount(
                        labels, minlength=n_classes
                    ).argmax()
        return output


def per_class_metrics(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-class precision, recall and F1 from a confusion matrix."""
    tp = np.diag(confusion).astype(float)
    precision_per_class = np.divide(
        tp, confusion.sum(axis=0), out=np.zeros_like(tp), where=confusion.sum(axis=0) != 0
    )
    recall_per_class = np.divide(
        tp, confusion.sum(axis=1), out=np.zeros_like(tp), where=confusion.sum(axis=1) != 0
    )
    f1_per_class = np.divide(
        2 * precision_per_class * recall_per_class,
        precision_per_class + recall_per_class,
        out=np.zeros_like(tp),
        where=(precision_per_class + recall_per_class) != 0,
    )
    return precision_per_class, recall_per_class, f1_per_class


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10
) -> tuple[float, float, float, float, np.ndarray]:
    """Compute accuracy and macro precision/recall/F1 from a confusion matrix."""
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    precision_per_class, recall_per_class, f1_per_class = per_class_metrics(confusion)
    return (
        float(np.trace(confusion) / confusion.sum()),
        float(precision_per_class.mean()),
        float(recall_per_class.mean()),
        float(f1_per_class.mean()),
        confusion,
    )


def cross_validate(
    x: np.ndarray,
    y: np.ndarray,
    k_values: list[int],
    distance_metrics: list[str],
    folds: int,
    batch_size: int,
    seed: int,
) -> tuple[list[Result], tuple[str, int], dict[str, int]]:
    split_indices = make_stratified_folds(y, folds, np.random.default_rng(seed))
    results: list[Result] = []

    for fold_number, validation_idx in enumerate(split_indices, start=1):
        train_idx = np.concatenate(
            [idx for i, idx in enumerate(split_indices, start=1) if i != fold_number]
        )
        for metric in distance_metrics:
            print(f"CV fold {fold_number}/{folds}: metric={metric}")
            started = time.perf_counter()
            model = KNearestNeighbors(metric, batch_size).fit(x[train_idx], y[train_idx])
            predictions = model.predict_many_k(x[validation_idx], k_values)
            elapsed = time.perf_counter() - started
            for k, y_pred in predictions.items():
                accuracy, precision, recall, f1, _ = classification_metrics(
                    y[validation_idx], y_pred
                )
                results.append(
                    Result("cross_validation", metric, k, fold_number, accuracy,
                           precision, recall, f1, elapsed)
                )

    # Select by mean macro-F1, then accuracy, then prefer smaller k on a tie.
    candidates: list[tuple[float, float, int, str, int]] = []
    for metric in distance_metrics:
        for k in k_values:
            rows = [r for r in results if r.metric == metric and r.k == k]
            candidates.append((
                float(np.mean([r.f1 for r in rows])),
                float(np.mean([r.accuracy for r in rows])),
                -k,
                metric,
                k,
            ))
    best_k_per_metric = {
        metric: max(c for c in candidates if c[3] == metric)[4] for metric in distance_metrics
    }
    best = max(candidates)
    return results, (best[3], best[4]), best_k_per_metric


def save_results(results: list[Result], output_dir: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys())
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def save_summary(results: list[Result], output_dir: Path) -> None:
    """Write mean and standard deviation over CV folds for each (metric, k)."""
    names = ("accuracy", "precision", "recall", "f1")
    cv = [r for r in results if r.stage == "cross_validation"]
    with (output_dir / "cv_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "k"] + [f"{n}_{s}" for n in names for s in ("mean", "std")])
        for metric, k in sorted({(r.metric, r.k) for r in cv}):
            rows = [r for r in cv if r.metric == metric and r.k == k]
            stats = []
            for name in names:
                values = [getattr(r, name) for r in rows]
                stats += [f"{np.mean(values):.4f}", f"{np.std(values):.4f}"]
            writer.writerow([metric, k] + stats)


def print_summary(results: list[Result]) -> None:
    print("\nMean 5-fold cross-validation results (macro averages)")
    print(f"{'metric':<12} {'k':>3} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    pairs = sorted({(r.metric, r.k) for r in results if r.stage == "cross_validation"})
    for metric, k in pairs:
        rows = [r for r in results if r.stage == "cross_validation" and r.metric == metric and r.k == k]
        values = [np.mean([getattr(r, name) for r in rows]) for name in
                  ("accuracy", "precision", "recall", "f1")]
        print(f"{metric:<12} {k:>3} " + " ".join(f"{value:>10.4f}" for value in values))

    tests = [r for r in results if r.stage == "test"]
    if tests:
        print("\nTest results (each metric with its best CV k; * = overall best)")
        for test in tests:
            mark = "*" if test.fold == "final" else " "
            print(f"{mark} metric={test.metric:<10} k={test.k:<3} accuracy={test.accuracy:.4f}, "
                  f"precision={test.precision:.4f}, recall={test.recall:.4f}, f1={test.f1:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 7, 9, 11, 15, 21])
    parser.add_argument("--metrics", nargs="+", choices=sorted(KNearestNeighbors.SUPPORTED_METRICS),
                        default=["euclidean", "manhattan", "chebyshev", "cosine"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=10000,
                        help="Stratified training subset size; 0 means all 50,000.")
    parser.add_argument("--max-test", type=int, default=2000,
                        help="Stratified test subset size; 0 means all 10,000.")
    parser.add_argument("--image-size", type=int, choices=[8, 16, 32], default=32,
                        help="Spatial size after average-pooling; 32 keeps original pixels.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    rng = np.random.default_rng(args.seed)
    raw_train, y_train, raw_test, y_test = load_cifar10(args.data_dir, args.download)
    raw_train, y_train = stratified_subset(raw_train, y_train, args.max_train, rng)
    raw_test, y_test = stratified_subset(raw_test, y_test, args.max_test, rng)
    x_train = preprocess(raw_train, args.image_size)
    x_test = preprocess(raw_test, args.image_size)
    print(f"Data: train={x_train.shape}, test={x_test.shape}")

    results, (best_metric, best_k), best_k_per_metric = cross_validate(
        x_train, y_train, args.k_values, args.metrics,
        args.folds, args.batch_size, args.seed,
    )
    print(f"Best CV setting: metric={best_metric}, k={best_k}")

    # Every metric is evaluated on the test set with its own best CV k for
    # comparison; the overall best CV setting is marked with fold="final".
    confusion = None
    for metric, k in best_k_per_metric.items():
        started = time.perf_counter()
        model = KNearestNeighbors(metric, args.batch_size).fit(x_train, y_train)
        y_pred = model.predict_many_k(x_test, [k])[k]
        accuracy, precision, recall, f1, matrix = classification_metrics(y_test, y_pred)
        is_best = metric == best_metric
        results.append(Result("test", metric, k, "final" if is_best else "per_metric",
                              accuracy, precision, recall, f1, time.perf_counter() - started))
        if is_best:
            confusion = matrix

    config = vars(args).copy()
    config.update(best_metric=best_metric, best_k=best_k, best_k_per_metric=best_k_per_metric,
                  train_shape=x_train.shape, test_shape=x_test.shape,
                  class_names=CLASS_NAMES)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    save_results(results, args.output_dir, config)
    save_summary(results, args.output_dir)
    np.savetxt(args.output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    precision_pc, recall_pc, f1_pc = per_class_metrics(confusion)
    with (args.output_dir / "per_class_test.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1"])
        for name, p, r, f in zip(CLASS_NAMES, precision_pc, recall_pc, f1_pc):
            writer.writerow([name, f"{p:.4f}", f"{r:.4f}", f"{f:.4f}"])
    print_summary(results)
    print(f"\nSaved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
