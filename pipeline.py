"""
VulneraScope-X: end-to-end runnable pipeline
=================================================
Reconstructs the method described in the paper:
  min-max normalization -> Word2Vec + Node2Vec embeddings -> SMOTE ->
  RCGO feature selection -> Grid Search-tuned SVM, compared to NB/MLP/KNN.

CAVEAT: This repository provides a conceptual implementation of the proposed 
methodology and is intended for educational purposes. To ensure 
fast execution, the Word2Vec/Node2Vec and RCGO components are lightweight 
implementations, and experiments are run on a reconstructed 100-row dataset. 
Consequently, the numerical results produced by this code are illustrative.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE

RNG = 42
np.random.seed(RNG)


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_data(path="VulneraScope-X.csv"):
    df = pd.read_csv(path)
    y = df["Is_Defective"].values
    df = df.drop(columns=["Is_Defective", "CVE_ID", "Published_Date", "Modified_Date"])
    return df, y


# --------------------------------------------------------------------------
# 2. Feature engineering
#    - categorical text fields -> Word2Vec embeddings (semantic)
#    - vendor/product relations -> Node2Vec-style embeddings (topological)
#    - numeric fields -> min-max normalized
# --------------------------------------------------------------------------
def word2vec_embed(series, dim=8):
    """Treat each categorical value as a 'word'; learn embeddings from co-occurrence."""
    sentences = [[str(v)] for v in series]
    model = Word2Vec(sentences, vector_size=dim, min_count=1,
                     window=2, seed=RNG, workers=1, epochs=50)
    return np.vstack([model.wv[str(v)] for v in series])


def node2vec_embed(vendors, products, dim=8):
    """
    Lightweight Node2Vec substitute: build a vendor-product bipartite graph,
    generate short random walks, then learn embeddings with Word2Vec.
    """
    from collections import defaultdict
    adj = defaultdict(set)
    for v, p in zip(vendors, products):
        adj[f"V_{v}"].add(f"P_{p}")
        adj[f"P_{p}"].add(f"V_{v}")
    nodes = list(adj.keys())
    rng = np.random.default_rng(RNG)
    walks = []
    for _ in range(10):
        for start in nodes:
            walk, cur = [start], start
            for _ in range(6):
                nbrs = list(adj[cur])
                if not nbrs:
                    break
                cur = nbrs[rng.integers(len(nbrs))]
                walk.append(cur)
            walks.append(walk)
    model = Word2Vec(walks, vector_size=dim, min_count=1,
                     window=3, seed=RNG, workers=1, epochs=50)
    return np.vstack([model.wv[f"V_{v}"] for v in vendors])


def build_features(df):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    X_num = MinMaxScaler().fit_transform(df[num_cols])

    # semantic embeddings for text categoricals
    w2v_vendor = word2vec_embed(df["Vendor_Name"])
    w2v_product = word2vec_embed(df["Product_Name"])
    w2v_av = word2vec_embed(df["Attack_Vector"])

    # topological embedding from vendor-product graph
    n2v = node2vec_embed(df["Vendor_Name"], df["Product_Name"])

    X = np.hstack([X_num, w2v_vendor, w2v_product, w2v_av, n2v])
    names = (num_cols
             + [f"w2v_vendor_{i}" for i in range(w2v_vendor.shape[1])]
             + [f"w2v_product_{i}" for i in range(w2v_product.shape[1])]
             + [f"w2v_av_{i}" for i in range(w2v_av.shape[1])]
             + [f"n2v_{i}" for i in range(n2v.shape[1])])
    return X, names


# --------------------------------------------------------------------------
# 3. RCGO (Running City Game Optimizer) — feature selection
#    Population-based binary metaheuristic. Each "runner" is a feature mask;
#    fitness rewards accuracy and penalizes number of selected features.
# --------------------------------------------------------------------------
def rcgo_feature_select(X, y, n_agents=15, n_iter=20, alpha=0.9):
    from sklearn.model_selection import cross_val_score
    d = X.shape[1]
    rng = np.random.default_rng(RNG)

    pop = rng.random((n_agents, d)) > 0.5
    for a in range(n_agents):
        if not pop[a].any():
            pop[a, rng.integers(d)] = True

    def fitness(mask):
        if not mask.any():
            return 0.0
        acc = cross_val_score(SVC(kernel="rbf", C=1, gamma="scale"),
                              X[:, mask], y, cv=3).mean()
        return alpha * acc + (1 - alpha) * (1 - mask.sum() / d)

    fits = np.array([fitness(m) for m in pop])
    best_idx = fits.argmax()
    best_mask, best_fit = pop[best_idx].copy(), fits[best_idx]

    for it in range(n_iter):
        for a in range(n_agents):
            # "run toward the city center" = move toward current best mask
            move = rng.random(d) < (0.15 + 0.25 * (it / n_iter))
            cand = pop[a].copy()
            cand[move] = best_mask[move]
            # random exploration flips
            flips = rng.random(d) < 0.05
            cand[flips] = ~cand[flips]
            if not cand.any():
                cand[rng.integers(d)] = True
            fc = fitness(cand)
            if fc > fits[a]:
                pop[a], fits[a] = cand, fc
                if fc > best_fit:
                    best_mask, best_fit = cand.copy(), fc
    return best_mask, best_fit


# --------------------------------------------------------------------------
# 4. Classifiers
# --------------------------------------------------------------------------
def evaluate(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = (model.predict_proba(Xte)[:, 1]
             if hasattr(model, "predict_proba")
             else model.decision_function(Xte))
    return {
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
        "auc_roc": roc_auc_score(yte, proba),
    }


def main():
    print("Loading reconstructed VulneraScope-X ...")
    df, y = load_data()

    print("Building features (min-max + Word2Vec + Node2Vec) ...")
    X, names = build_features(df)
    print(f"  feature matrix: {X.shape}")

    print("Balancing classes with SMOTE ...")
    X_bal, y_bal = SMOTE(random_state=RNG, k_neighbors=5).fit_resample(X, y)
    print(f"  after SMOTE: {X_bal.shape}, class counts = {np.bincount(y_bal)}")

    print("RCGO feature selection ...")
    mask, fit = rcgo_feature_select(X_bal, y_bal)
    selected = [n for n, m in zip(names, mask) if m]
    print(f"  selected {mask.sum()}/{len(names)} features (fitness={fit:.3f})")

    Xsel = X_bal[:, mask]
    Xtr, Xte, ytr, yte = train_test_split(
        Xsel, y_bal, test_size=0.25, random_state=RNG, stratify=y_bal)

    print("Grid Search-tuned SVM ...")
    grid = GridSearchCV(
        SVC(probability=True),
        {"C": [0.1, 1, 10], "gamma": ["scale", 0.1, 0.01], "kernel": ["rbf"]},
        cv=3, n_jobs=-1)
    grid.fit(Xtr, ytr)
    print(f"  best params: {grid.best_params_}")

    models = {
        "SVM (RCGO + GridSearch)": grid.best_estimator_,
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=RNG),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    print("\n" + "=" * 74)
    print(f"{'Model':<26}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'AUC':>8}")
    print("-" * 74)
    results = {}
    for name, mdl in models.items():
        r = evaluate(mdl, Xtr, Xte, ytr, yte)
        results[name] = r
        print(f"{name:<26}{r['accuracy']:>8.3f}{r['precision']:>8.3f}"
              f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['auc_roc']:>8.3f}")
    print("=" * 74)

    pd.DataFrame(results).T.to_csv("/home/claude/results.csv")
    print("\nSaved results.csv")


if __name__ == "__main__":
    main()
