import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import scipy.io

from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay)
from sklearn.model_selection import (train_test_split,StratifiedKFold)

from sklearn.preprocessing import (StandardScaler,MinMaxScaler)

from sklearn.pipeline import Pipeline


# CONFIG


DATA_DIR = Path(".")
RANDOM_STATE = 42
VAL_SIZE = 0.2
AE_MAX_ITER = 200
N_SPLITS = 5

DATASETS = {"Speech": {"file": "speech.mat"}, "Mammography": {"file": "mammography.mat"},"Optdigits": {"file": "optdigits.mat"}}


# DATA LOADING


def load_mat_v5(path):

    mat = scipy.io.loadmat(str(path))

    X = mat["X"].astype(float)
    y = mat["y"].ravel().astype(int)

    return X, y


def load_dataset(name, cfg):

    path = DATA_DIR / cfg["file"]
    X, y = load_mat_v5(path)
    print(f"[{name}] "f"{len(X):,} rows | "f"{X.shape[1]} features | "f"{y.mean()*100:.2f}% anomalies")
    return X, y



# METRICS


def determine_threshold(val_scores, contamination):
    return np.quantile(val_scores, 1 - contamination,)

def compute_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    return {"AUROC": roc_auc_score(y_true, y_score),
        "AP":average_precision_score(y_true, y_score),
        "Precision":precision_score(y_true, y_pred, zero_division=0),
        "Recall":recall_score(y_true, y_pred, zero_division=0),
        "F1":f1_score(y_true,y_pred, zero_division=0),
        "Confusion":confusion_matrix(y_true,y_pred),
    }



# IQR


def method_iqr(X_train,X_val,X_test,y_test,contamination):

    Q1 = np.percentile(X_train,25,axis=0)
    Q3 = np.percentile(X_train,75,axis=0)

    IQR = np.where(Q3 - Q1 == 0, 1e-9, Q3 - Q1,)

    def score(X):
        below = (np.maximum(Q1 - 1.5 * IQR- X, 0) / IQR)
        above = (np.maximum(X- (Q3+ 1.5 * IQR),0)/ IQR)
        return (below + above).mean(axis=1)

    val_scores = score(X_val)
    threshold = determine_threshold(val_scores, contamination)

    test_scores = score(X_test)

    return compute_metrics(y_test,test_scores,threshold)



# ISOLATION FOREST


def method_iforest(X_train,X_val,X_test,y_test,contamination):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ( "model", IsolationForest(n_estimators=200,contamination=contamination,random_state=RANDOM_STATE,n_jobs=-1))])

    pipe.fit(X_train)

    val_scores = (-pipe.decision_function(X_val))

    threshold = determine_threshold(val_scores,contamination)

    test_scores = (-pipe.decision_function(X_test))

    metrics = compute_metrics(y_test,test_scores,threshold)

    return metrics, test_scores



# AUTOENCODER


def method_autoencoder(X_train,y_train,X_val,X_test,y_test,contamination):

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_normal = X_train[y_train == 0]
    d = X_train.shape[1]
    latent = max(2,d // 4,)

    hidden = (
        max(latent * 4, 32),
        max(latent * 2, 16),
        latent,
        max(latent * 2, 16),
        max(latent * 4, 32),
    )

    model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", max_iter=AE_MAX_ITER, random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1)
    model.fit(X_normal, X_normal)

    def mse(X):
        return np.mean((X- model.predict(X)) ** 2,axis=1)

    val_scores = mse(X_val)

    threshold = determine_threshold(val_scores,contamination)

    test_scores = mse(X_test)

    metrics = compute_metrics(y_test,test_scores,threshold)

    return metrics, test_scores



# ENSEMBLE


def method_ensemble(if_scores,ae_scores,y_test,contamination):

    def normalize(s):
        return (s - s.min()) / (s.max() - s.min()+ 1e-9)

    if_norm = normalize(if_scores)
    ae_norm = normalize(ae_scores)

    combined = (0.5 * if_norm+ 0.5 * ae_norm)

    threshold = determine_threshold(combined,contamination)

    return compute_metrics(y_test,combined,threshold)



# RUN


all_results = {}

for ds_name, ds_cfg in DATASETS.items():

    print()
    print("=" * 60)
    print(ds_name)
    print("=" * 60)

    X, y = load_dataset(ds_name,ds_cfg)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):

        print("Fold:", fold + 1)

        X_train_full = X[train_idx]
        y_train_full = y[train_idx]

        X_test = X[test_idx]
        y_test = y[test_idx]

        X_train, X_val, y_train, y_val = train_test_split(X_train_full,y_train_full,test_size=VAL_SIZE,stratify=y_train_full,random_state=RANDOM_STATE,)

        contamination = np.clip(y_train.mean(),0.01, 0.4)

        res_iqr = method_iqr(X_train,X_val,X_test,y_test,contamination)

        res_if, if_scores = method_iforest(X_train,X_val, X_test,y_test,contamination)

        res_ae, ae_scores = method_autoencoder(X_train,y_train,X_val,X_test,y_test,contamination)

        res_ens = method_ensemble(if_scores,ae_scores, y_test,contamination)

        fold_results.append({
            "IQR": res_iqr,
            "IForest": res_if,
            "Autoencoder": res_ae,
            "IF+AE": res_ens,
        })

    all_results[ds_name] = fold_results



# AGGREGATE RESULTS


METHODS = ["IQR", "IForest", "Autoencoder", "IF+AE"]

metrics_list = ["AUROC","AP","Precision","Recall","F1",]

rows = []

for ds, folds in all_results.items():
    for method in METHODS:
        for metric in metrics_list:
            values = [fold[method][metric] for fold in folds]
            mean_val = np.mean(values)
            rows.append({"Datensatz": ds,"Methode": method,"Metrik": metric,"Wert": mean_val})

results_df = pd.DataFrame(rows)

results_df.to_csv("anomaly_results_summary.csv",index=False,)

print("→ anomaly_results_summary.csv gespeichert")



# PLOT 1 – BARS


DS_NAMES = list(all_results.keys())

fig, axes = plt.subplots(3,1,figsize=(11, 12),)

for ax, metric in zip(axes,["AUROC", "AP", "F1"],):
    x = np.arange(len(DS_NAMES))
    bw = 0.2
    for i, method in enumerate(METHODS):
        vals = []
        for ds in DS_NAMES:
            values = [fold[method][metric] for fold in all_results[ds]]
            vals.append(np.mean(values))

        ax.bar(x + i * bw,vals,bw,label=method,alpha=0.85,)

    ax.set_title(metric)
    ax.set_xticks(x + bw * 1.5)
    ax.set_xticklabels(DS_NAMES)
    ax.set_ylim(0,1.05,)
    ax.legend()
    ax.grid(axis="y",alpha=0.3,)

plt.tight_layout()

plt.savefig("plot1_bars.png",dpi=150,)

plt.close()

print("→ plot1_bars.png gespeichert")



# PLOT 2 – HEATMAP


heat = pd.DataFrame({m: [np.mean([fold[m]["AUROC"]for fold in all_results[ds]])
            for ds in DS_NAMES]
        for m in METHODS},index=DS_NAMES,)

fig, ax = plt.subplots(figsize=(7, 4),)

sns.heatmap(heat,annot=True,fmt=".3f",cmap="RdYlGn",vmin=0,vmax=1,ax=ax)

ax.set_title("AUROC – alle Methoden",)

plt.tight_layout()

plt.savefig("plot2_heatmap.png",dpi=150,)

plt.close()

print("→ plot2_heatmap.png gespeichert")



# PLOT 3 – CONFUSION MATRICES


for ds in DS_NAMES:
    fig, axes_cm = plt.subplots(1,len(METHODS),figsize=(16, 4),)

    fig.suptitle(f"Konfusionsmatrizen – {ds}", fontweight="bold",)

    for ax, method in zip(axes_cm,METHODS,):

        cm_total = np.zeros((2, 2),dtype=int,  )
        for fold in all_results[ds]:
            cm_total += fold[method]["Confusion"]

        cm_total = cm_total.astype(int)

        ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=["Normal","Anomalie"]).plot(ax=ax,colorbar=False,cmap="Blues",values_format="d")
        ax.set_title(method)

    plt.tight_layout()

    fname = f"plot3_cm_{ds}.png"

    plt.savefig(fname,dpi=150,bbox_inches="tight")

    plt.close()

    print("→", fname, "gespeichert")

print()
print("Fertig")
print("Ausgabedateien:")
print("- anomaly_results_summary.csv")
print("- plot1_bars.png")
print("- plot2_heatmap.png")
print("- plot3_cm_<Datensatz>.png")