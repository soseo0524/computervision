"""
Linear Classification on CIFAR-10 with randomly initialized weights (no training).

Tasks
  1. Initialize W (small random values) and b (zeros) with a fixed seed
  2. Scores f(x) = Wx + b and predicted class for several images
  3. Multiclass SVM hinge loss (average over the selected examples)
  4. Softmax: compare scores with probabilities, check sum == 1
  5. Correctly classified / misclassified examples
  6. Visualize each class' weight vector as a 32x32x3 image
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar-10-batches-py")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
NUM_CLASSES = 10
NUM_SELECTED = 10      # images used for the score tables / average losses
SEARCH_RANGE = 1000    # test images scanned to find correct / wrong examples
DELTA = 1.0            # SVM margin

log_lines = []
def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    log_lines.append(s)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_batch(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    return d[b"data"], np.array(d[b"labels"])

def load_cifar10(data_dir):
    xs, ys = [], []
    for i in range(1, 6):
        x, y = load_batch(os.path.join(data_dir, f"data_batch_{i}"))
        xs.append(x); ys.append(y)
    x_train = np.concatenate(xs).astype(np.float64)
    x_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    with open(os.path.join(data_dir, "batches.meta"), "rb") as f:
        names = [n.decode() for n in pickle.load(f, encoding="bytes")[b"label_names"]]
    return x_train, x_test.astype(np.float64), y_test, names

def to_image(row):
    """flat CIFAR row (3072, order R,G,B planes) -> (32, 32, 3) uint8 image"""
    return row.reshape(3, 32, 32).transpose(1, 2, 0).astype(np.uint8)


# ----------------------------------------------------------------------------
# Model pieces
# ----------------------------------------------------------------------------
def init_params(num_classes, dim, seed=SEED, scale=0.01):
    rng = np.random.default_rng(seed)
    W = scale * rng.standard_normal((num_classes, dim))   # small random values
    b = np.zeros(num_classes)                             # bias = 0
    return W, b

def scores_fn(X, W, b):
    """f(x) = Wx + b  (X: N x D -> scores: N x C)"""
    return X @ W.T + b

def svm_loss_per_example(scores, y, delta=DELTA):
    """L_i = sum_{j != y_i} max(0, s_j - s_{y_i} + delta)"""
    N = scores.shape[0]
    correct = scores[np.arange(N), y][:, None]
    margins = np.maximum(0.0, scores - correct + delta)
    margins[np.arange(N), y] = 0.0
    return margins.sum(axis=1), margins

def svm_loss_loop(scores, y, delta=DELTA):
    """Naive loop version, used only to cross-check the vectorized one."""
    out = []
    for i in range(scores.shape[0]):
        li = 0.0
        for j in range(scores.shape[1]):
            if j == y[i]:
                continue
            li += max(0.0, scores[i, j] - scores[i, y[i]] + delta)
        out.append(li)
    return np.array(out)

def softmax(scores):
    """Numerically stable softmax over classes (rows)."""
    z = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def softmax_loss_per_example(probs, y):
    return -np.log(probs[np.arange(len(y)), y])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    x_train, x_test_raw, y_test, names = load_cifar10(DATA_DIR)
    log(f"Loaded CIFAR-10: train {x_train.shape}, test {x_test_raw.shape}")

    # Preprocessing: scale to [0,1] and subtract the training mean image
    mean_img = (x_train / 255.0).mean(axis=0)
    X_test = x_test_raw / 255.0 - mean_img
    D = X_test.shape[1]

    # 1) Parameters ------------------------------------------------------
    W, b = init_params(NUM_CLASSES, D)
    log(f"\n[1] Init: seed={SEED}, W shape={W.shape}, W mean={W.mean():+.5f}, "
        f"W std={W.std():.5f}, W min/max={W.min():+.4f}/{W.max():+.4f}")
    log(f"    b = {b}")

    # 2) Scores on selected images -------------------------------------
    idx = np.arange(NUM_SELECTED)
    Xs, ys = X_test[idx], y_test[idx]
    S = scores_fn(Xs, W, b)
    pred = S.argmax(axis=1)
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    log(f"\n[2] Scores f(x)=Wx+b for the first {NUM_SELECTED} test images")
    header = "img  true        pred        " + " ".join(f"{n[:6]:>8s}" for n in names)
    log(header)
    for k in range(NUM_SELECTED):
        row = " ".join(f"{v:8.4f}" for v in S[k])
        log(f"{idx[k]:<4d} {names[ys[k]]:<11s} {names[pred[k]]:<11s} {row}")
    log(f"    -> correct: {(pred == ys).sum()} / {NUM_SELECTED}")

    # 3) SVM hinge loss --------------------------------------------------
    Li, margins = svm_loss_per_example(S, ys)
    assert np.allclose(Li, svm_loss_loop(S, ys)), "vectorized vs loop SVM mismatch"
    log(f"\n[3] Multiclass SVM loss (delta={DELTA})")
    for k in range(NUM_SELECTED):
        log(f"    img {idx[k]:<3d} L_i = {Li[k]:.4f}")
    log(f"    average SVM loss = {Li.mean():.4f}   (expected ~ (C-1)*delta = {(NUM_CLASSES-1)*DELTA:.1f} "
        f"for near-equal scores)")

    # 4) Softmax ---------------------------------------------------------
    P = softmax(S)
    sums = P.sum(axis=1)
    Lsm = softmax_loss_per_example(P, ys)
    log("\n[4] Softmax probabilities (rows = images)")
    log("img  " + " ".join(f"{n[:6]:>8s}" for n in names) + "    sum")
    for k in range(NUM_SELECTED):
        log(f"{idx[k]:<4d} " + " ".join(f"{v:8.4f}" for v in P[k]) + f"  {sums[k]:.6f}")
    log(f"    all sums == 1 : {np.allclose(sums, 1.0)}   (max |sum-1| = {np.abs(sums-1).max():.2e})")
    log(f"    argmax(scores) == argmax(probs): {(S.argmax(1) == P.argmax(1)).all()}")
    log(f"    average softmax (cross-entropy) loss = {Lsm.mean():.4f}   (ln(10) = {np.log(10):.4f})")

    # 5) Correct / wrong examples ---------------------------------------
    S_all = scores_fn(X_test[:SEARCH_RANGE], W, b)
    pred_all = S_all.argmax(axis=1)
    ok = np.where(pred_all == y_test[:SEARCH_RANGE])[0]
    bad = np.where(pred_all != y_test[:SEARCH_RANGE])[0]
    log(f"\n[5] Among the first {SEARCH_RANGE} test images: {len(ok)} correct, {len(bad)} wrong "
        f"(chance level ~ 10%)")
    show_ok, show_bad = ok[:4], bad[:4]
    P_all = softmax(S_all)

    # ---------------- Figures ------------------------------------------
    # Fig 1: selected images with predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.6))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(to_image(x_test_raw[idx[k]])); ax.axis("off")
        good = pred[k] == ys[k]
        ax.set_title(f"#{idx[k]}  true: {names[ys[k]]}\npred: {names[pred[k]]}",
                     fontsize=9, color="green" if good else "red")
    fig.suptitle("Selected test images: true label vs. predicted label (random W)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig1_selected_images.png"), dpi=150); plt.close(fig)

    # Fig 2: scores vs softmax probabilities for 4 images
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for c, k in enumerate(range(4)):
        a0, a1 = axes[0, c], axes[1, c]
        cols = ["tab:gray"] * NUM_CLASSES
        cols[ys[k]] = "tab:green"
        if pred[k] != ys[k]:
            cols[pred[k]] = "tab:red"
        a0.bar(range(NUM_CLASSES), S[k], color=cols)
        a0.set_title(f"img #{idx[k]} (true: {names[ys[k]]})\nraw scores", fontsize=9)
        a1.bar(range(NUM_CLASSES), P[k], color=cols)
        a1.axhline(0.1, color="k", ls="--", lw=0.8)
        a1.set_title(f"softmax probs (sum={P[k].sum():.4f})", fontsize=9)
        for a in (a0, a1):
            a.set_xticks(range(NUM_CLASSES)); a.set_xticklabels(names, rotation=60, fontsize=7)
    fig.suptitle("Raw scores vs. Softmax probabilities (green=true, red=wrong prediction, dashed=uniform 0.1)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig2_scores_vs_softmax.png"), dpi=150); plt.close(fig)

    # Fig 3: correct and wrong examples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for row, (grp, title, colr) in enumerate([(show_ok, "Correct", "green"), (show_bad, "Wrong", "red")]):
        for c in range(4):
            ax = axes[row, c]; ax.axis("off")
            if c < len(grp):
                i = grp[c]
                ax.imshow(to_image(x_test_raw[i]))
                ax.set_title(f"{title} #{i}\ntrue: {names[y_test[i]]} / pred: {names[pred_all[i]]}\n"
                             f"p(pred)={P_all[i].max():.3f}  p(true)={P_all[i][y_test[i]]:.3f}",
                             fontsize=8, color=colr)
    fig.suptitle("Correctly classified vs. misclassified examples (random W)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig3_correct_wrong.png"), dpi=150); plt.close(fig)
    for i in show_ok:
        log(f"    correct  #{i}: true={names[y_test[i]]}, pred={names[pred_all[i]]}, "
            f"p(pred)={P_all[i].max():.4f}")
    for i in show_bad:
        log(f"    wrong    #{i}: true={names[y_test[i]]}, pred={names[pred_all[i]]}, "
            f"p(pred)={P_all[i].max():.4f}, p(true)={P_all[i][y_test[i]]:.4f}")

    # 6) Weight visualization -------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.6))
    for c, ax in enumerate(axes.ravel()):
        w = W[c].reshape(3, 32, 32).transpose(1, 2, 0)
        w_img = (255.0 * (w - w.min()) / (w.max() - w.min())).astype(np.uint8)
        ax.imshow(w_img); ax.axis("off"); ax.set_title(names[c], fontsize=10)
    fig.suptitle("Random (untrained) weight vector of each class as 32x32x3 image (min-max normalized)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "fig4_weights.png"), dpi=150); plt.close(fig)

    # quantitative support for the explanation: correlation between class templates
    Wc = np.corrcoef(W)
    off = Wc[~np.eye(NUM_CLASSES, dtype=bool)]
    log(f"\n[6] Weight images saved. Pairwise correlation between class weight vectors: "
        f"mean |corr| = {np.abs(off).mean():.4f}, max |corr| = {np.abs(off).max():.4f} "
        f"(~1/sqrt(3072) = {1/np.sqrt(D):.4f} expected for independent noise)")
    # spatial smoothness: correlation of neighbouring pixels in W vs in a real image
    w0 = W[0].reshape(3, 32, 32)
    nb_w = np.corrcoef(w0[:, :, :-1].ravel(), w0[:, :, 1:].ravel())[0, 1]
    im0 = (x_test_raw[0] / 255.0).reshape(3, 32, 32)
    nb_i = np.corrcoef(im0[:, :, :-1].ravel(), im0[:, :, 1:].ravel())[0, 1]
    log(f"    neighbouring-pixel correlation: weights(class 0) = {nb_w:+.3f}, real image #0 = {nb_i:+.3f}")

    with open(os.path.join(OUT_DIR, "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

if __name__ == "__main__":
    main()
