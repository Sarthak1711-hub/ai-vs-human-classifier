# =====================================================
# Install Libraries
# =====================================================
# !pip install -U transformers datasets scikit-learn matplotlib seaborn pandas

# =====================================================
# Imports
# =====================================================
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)

# =====================================================
# GPU Check
# =====================================================
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# =====================================================
# Configuration  ← single source of truth
# =====================================================
TOTAL_SAMPLES = 900       # rows sampled from CSV
TRAIN_SIZE    = 720       # 80 % of 900
VAL_SIZE      = 180       # 20 % of 900
RANDOM_STATE  = 42
MAX_LENGTH    = 128
EPOCHS        = 2
BATCH_SIZE    = 4
LEARNING_RATE = 2e-5

# =====================================================
# Load Dataset
# =====================================================
df_raw = pd.read_csv("AI_Human.csv", engine="python", on_bad_lines="skip")
df_raw = df_raw.rename(columns={"generated": "label"})
df_raw["label"] = df_raw["label"].astype(int)

df = df_raw.sample(n=TOTAL_SAMPLES, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"\n{'='*45}")
print(f"  Dataset : {TOTAL_SAMPLES} samples total")
print(f"  Train   : {TRAIN_SIZE}  |  Val : {VAL_SIZE}")
print(f"{'='*45}")
print(df["label"].value_counts().rename({0: "Human (0)", 1: "AI (1)"}).to_string())
print(f"{'='*45}\n")

# =====================================================
# EDA Visualizations
# =====================================================
df["length"] = df["text"].apply(lambda x: len(str(x).split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Exploratory Data Analysis  (n=900)", fontsize=15, fontweight="bold")

counts = df["label"].value_counts().rename({0: "Human", 1: "AI"})
axes[0].bar(
    counts.index, counts.values,
    color=["#2196F3", "#FF5722"], edgecolor="black", width=0.5,
)
axes[0].set_title("Class Distribution")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Count")
axes[0].set_xticks(range(len(counts)))
axes[0].set_xticklabels(counts.index)
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 4, str(v), ha="center", fontweight="bold")

label_map = df["label"].map({0: "Human", 1: "AI"})
sns.kdeplot(
    data=df, x="length", hue=label_map, fill=True,
    ax=axes[1], palette={"Human": "#2196F3", "AI": "#FF5722"}, alpha=0.5,
)
axes[1].set_title("Text Length Distribution (words)")
axes[1].set_xlabel("Word Count")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()

# =====================================================
# Build HuggingFace Dataset
# =====================================================
hf_dataset = Dataset.from_pandas(df[["text", "label"]])
hf_dataset = hf_dataset.train_test_split(
    test_size=VAL_SIZE / TOTAL_SAMPLES, seed=RANDOM_STATE
)

assert len(hf_dataset["train"]) == TRAIN_SIZE, \
    f"Train mismatch: {len(hf_dataset['train'])} ≠ {TRAIN_SIZE}"
assert len(hf_dataset["test"]) == VAL_SIZE, \
    f"Val mismatch: {len(hf_dataset['test'])} ≠ {VAL_SIZE}"

print(f"Train set : {len(hf_dataset['train'])} samples")
print(f"Val   set : {len(hf_dataset['test'])} samples\n")

# =====================================================
# Tokenization Helper
# =====================================================
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

# =====================================================
# Metrics
# =====================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# =====================================================
# Storage
# =====================================================
all_results = []   # list[dict]  → final comparison table
roc_data    = {}   # short_name → (fpr, tpr, roc_auc)
conf_mats   = {}   # short_name → ndarray

# =====================================================
# Training Function
# =====================================================
def train_model(model_name: str) -> None:
    gc.collect()
    torch.cuda.empty_cache()

    short_name = model_name.split("/")[-1]
    print(f"\n{'='*50}")
    print(f"  Training : {short_name}")
    print(f"{'='*50}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = hf_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=500,
    )
    tokenized = tokenized.remove_columns(["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    train_ds = tokenized["train"]   # 720
    val_ds   = tokenized["test"]    # 180

    assert len(train_ds) == TRAIN_SIZE
    assert len(val_ds)   == VAL_SIZE

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir=f"./results_{short_name}",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Predictions
    preds_output = trainer.predict(val_ds)
    logits = preds_output.predictions
    preds  = np.argmax(logits, axis=1)
    labels = preds_output.label_ids

    # Confusion matrix
    conf_mats[short_name] = confusion_matrix(labels, preds)

    # ROC / AUC
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    roc_data[short_name] = (fpr, tpr, roc_auc)

    # Metrics
    eval_results = trainer.evaluate()
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    all_results.append(
        {
            "Model"    : short_name,
            "Accuracy" : round(eval_results["eval_accuracy"], 4),
            "Precision": round(float(p),  4),
            "Recall"   : round(float(r),  4),
            "F1"       : round(eval_results["eval_f1"], 4),
            "AUC-ROC"  : round(roc_auc, 4),
        }
    )

    print(
        f"\n  ✅ {short_name}  →  "
        f"Acc={eval_results['eval_accuracy']:.4f}  "
        f"F1={eval_results['eval_f1']:.4f}  "
        f"AUC={roc_auc:.4f}"
    )

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

# =====================================================
# Models
# =====================================================
MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "distilbert-base-uncased",
    "albert-base-v2",
    "xlnet-base-cased",
]

# =====================================================
# Train Loop
# =====================================================
for m in MODELS:
    train_model(m)

# =====================================================
# Results DataFrame  (keep insertion order for plots,
# sort by F1 only for the printed ranking table)
# =====================================================
df_results = pd.DataFrame(all_results)           # original training order
df_ranked  = (
    df_results
    .sort_values("F1", ascending=False)
    .reset_index(drop=True)
)
df_ranked.index += 1    # rank starts at 1

# =====================================================
# Console: pretty comparison table
# =====================================================
SEP = "=" * 75
print(f"\n{SEP}")
print("  MODEL COMPARISON TABLE  (900 samples | 720 train / 180 val)")
print(SEP)
print(
    f"{'Rank':<6}{'Model':<26}"
    f"{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'AUC':>8}"
)
print("-" * 75)
for rank, row in df_ranked.iterrows():
    print(
        f"{rank:<6}{row['Model']:<26}"
        f"{row['Accuracy']:>8.4f}{row['Precision']:>8.4f}"
        f"{row['Recall']:>8.4f}{row['F1']:>8.4f}{row['AUC-ROC']:>8.4f}"
    )
print(SEP)
print(f"\n  Best model : {df_ranked.iloc[0]['Model']}"
      f"  (F1 = {df_ranked.iloc[0]['F1']:.4f})\n")

# =====================================================
# Plot 1 — Confusion Matrices  (1 row × 5 cols)
# =====================================================
n_models = len(conf_mats)
fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
fig.suptitle(
    "Confusion Matrices  (val set, n=180)",
    fontsize=14, fontweight="bold",
)

for ax, (name, cm) in zip(axes, conf_mats.items()):
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Human", "AI"], yticklabels=["Human", "AI"],
        cbar=False, linewidths=0.5,
    )
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()

# =====================================================
# Plot 2 — Grouped Bar: all 5 metrics per model
#           (use df_results — insertion / training order)
# =====================================================
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
colors  = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
x       = np.arange(len(df_results))
width   = 0.14

fig, ax = plt.subplots(figsize=(16, 6))
for i, (metric, color) in enumerate(zip(metrics, colors)):
    offset = (i - len(metrics) / 2 + 0.5) * width
    bars = ax.bar(
        x + offset, df_results[metric], width,
        label=metric, color=color, edgecolor="white", linewidth=0.5,
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.004,
            f"{h:.3f}", ha="center", va="bottom", fontsize=7, rotation=90,
        )

ax.set_xticks(x)
ax.set_xticklabels(df_results["Model"], rotation=15, ha="right", fontsize=10)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title(
    "Model Comparison — All Metrics  (900 samples | 720 train / 180 val)",
    fontsize=13, fontweight="bold",
)
ax.legend(loc="lower right", fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig("metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# =====================================================
# Plot 3 — ROC Curves
# =====================================================
line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
fig, ax = plt.subplots(figsize=(8, 7))

for (name, (fpr, tpr, roc_auc)), ls in zip(roc_data.items(), line_styles):
    ax.plot(fpr, tpr, lw=2, linestyle=ls,
            label=f"{name}  (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison  (900 samples)", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# =====================================================
# Plot 4 — Line Plot: Accuracy vs F1  (training order)
# =====================================================
model_names = df_results["Model"].tolist()   # consistent x-axis
x_pos       = range(len(model_names))        # integer positions → no string-index misalign

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_pos, df_results["Accuracy"], marker="o", lw=2, ms=8,
        label="Accuracy", color="#2196F3")
ax.plot(x_pos, df_results["F1"],       marker="s", lw=2, ms=8,
        label="F1 Score",  color="#F44336")

for i, row in df_results.iterrows():
    ax.annotate(
        f"{row['Accuracy']:.3f}", (i, row["Accuracy"]),
        textcoords="offset points", xytext=(0, 9),
        ha="center", fontsize=9, color="#2196F3",
    )
    ax.annotate(
        f"{row['F1']:.3f}", (i, row["F1"]),
        textcoords="offset points", xytext=(0, -15),
        ha="center", fontsize=9, color="#F44336",
    )

ax.set_xticks(list(x_pos))
ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=10)
ax.set_ylim(0.5, 1.08)
ax.set_ylabel("Score")
ax.set_title(
    "Accuracy vs F1 Score per Model  (900 samples)",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("accuracy_f1_line.png", dpi=150, bbox_inches="tight")
plt.show()