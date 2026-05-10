import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, mean_absolute_error, r2_score
)
import os
import warnings
warnings.filterwarnings("ignore")

# ── Colours matching MakerPlan UI ────────────────────────────────────────────
NAVY   = "#0f2044"
ACCENT = "#4f8ef7"
GREEN  = "#22c55e"
RED    = "#ef4444"
YELLOW = "#f59e0b"

print("=" * 60)
print("  MAKERPLAN — ML MODEL EVALUATION REPORT")
print("=" * 60)

# ── Load dataset ─────────────────────────────────────────────────────────────
df = pd.read_csv("Maker-Dataset.csv")
print(f"\nDataset loaded: {len(df)} records")
print(f"Columns: {list(df.columns)}")

# ── Load encoders ─────────────────────────────────────────────────────────────
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# ── Encode features ───────────────────────────────────────────────────────────
for col in ['product_idea', 'category', 'skill_level',
            'available_materials', 'available_tools', 'build_feasible']:
    df[col + '_enc'] = encoders[col].transform(df[col].astype(str))

features = ['skill_level_enc', 'category_enc']
X = df[features]

# ── Recreate train/test splits with same random_state ────────────────────────
_, X_test_cost, _, y_test_cost = train_test_split(
    X, df['estimated_cost_zar'], test_size=0.2, random_state=42)

_, X_test_feas, _, y_test_feas = train_test_split(
    X, df['build_feasible_enc'], test_size=0.2, random_state=42)

_, X_test_skill, _, y_test_skill = train_test_split(
    X, df['skill_level_enc'], test_size=0.2, random_state=42)

# ── Load models ───────────────────────────────────────────────────────────────
with open("cost_model.pkl",  "rb") as f: cost_model  = pickle.load(f)
with open("feas_model.pkl",  "rb") as f: feas_model  = pickle.load(f)
with open("skill_model.pkl", "rb") as f: skill_model = pickle.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — COST PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODEL 1: Cost Predictor (Random Forest Regressor)")
print("─" * 60)

y_pred_cost = cost_model.predict(X_test_cost)
mae  = mean_absolute_error(y_test_cost, y_pred_cost)
r2   = r2_score(y_test_cost, y_pred_cost)
rmse = np.sqrt(np.mean((y_test_cost - y_pred_cost) ** 2))

print(f"  Mean Absolute Error (MAE) : R{mae:.2f}")
print(f"  Root Mean Squared Error   : R{rmse:.2f}")
print(f"  R² Score                  : {r2:.4f}")
print(f"  Test samples              : {len(y_test_cost)}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — FEASIBILITY PREDICTOR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODEL 2: Feasibility Predictor (Random Forest Classifier)")
print("─" * 60)

y_pred_feas = feas_model.predict(X_test_feas)
acc_feas = accuracy_score(y_test_feas, y_pred_feas)
feas_labels = encoders['build_feasible'].classes_

print(f"  Accuracy     : {acc_feas * 100:.1f}%")
print(f"  Test samples : {len(y_test_feas)}")
print(f"\n  Classification Report:")
print(classification_report(
    y_test_feas, y_pred_feas,
    target_names=feas_labels
))

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — COMPLEXITY CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  MODEL 3: Complexity Classifier (Random Forest Classifier)")
print("─" * 60)

y_pred_skill = skill_model.predict(X_test_skill)
acc_skill = accuracy_score(y_test_skill, y_pred_skill)
skill_labels = encoders['skill_level'].classes_

print(f"  Accuracy     : {acc_skill * 100:.1f}%")
print(f"  Test samples : {len(y_test_skill)}")
print(f"\n  Classification Report:")
print(classification_report(
    y_test_skill, y_pred_skill,
    target_names=skill_labels
))

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("  Generating evaluation charts...")
print("─" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor("#f8fafc")
fig.suptitle(
    "MakerPlan — ML Model Evaluation Report",
    fontsize=16, fontweight="bold", color=NAVY, y=0.98
)

# ── Chart 1: Actual vs Predicted Cost ────────────────────────────────────────
ax1 = axes[0, 0]
ax1.set_facecolor("white")
ax1.scatter(y_test_cost, y_pred_cost,
            color=ACCENT, alpha=0.6, edgecolors=NAVY, linewidth=0.5, s=40)
min_val = min(y_test_cost.min(), y_pred_cost.min())
max_val = max(y_test_cost.max(), y_pred_cost.max())
ax1.plot([min_val, max_val], [min_val, max_val],
         color=RED, linewidth=2, linestyle="--", label="Perfect prediction")
ax1.set_title("Model 1: Actual vs Predicted Cost",
              fontweight="bold", color=NAVY, pad=12)
ax1.set_xlabel("Actual Cost (ZAR)", color=NAVY)
ax1.set_ylabel("Predicted Cost (ZAR)", color=NAVY)
ax1.legend(fontsize=9)
ax1.text(0.05, 0.92, f"MAE: R{mae:.2f}  |  R²: {r2:.3f}",
         transform=ax1.transAxes, fontsize=9,
         color="white", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=NAVY, alpha=0.85))
ax1.grid(True, alpha=0.3)
ax1.spines[['top', 'right']].set_visible(False)

# ── Chart 2: Feasibility Confusion Matrix ────────────────────────────────────
ax2 = axes[0, 1]
ax2.set_facecolor("white")
cm_feas = confusion_matrix(y_test_feas, y_pred_feas)
im = ax2.imshow(cm_feas, cmap="Blues", aspect="auto")
ax2.set_xticks(range(len(feas_labels)))
ax2.set_yticks(range(len(feas_labels)))
ax2.set_xticklabels(feas_labels, color=NAVY)
ax2.set_yticklabels(feas_labels, color=NAVY)
for i in range(len(feas_labels)):
    for j in range(len(feas_labels)):
        ax2.text(j, i, str(cm_feas[i, j]),
                 ha="center", va="center",
                 fontsize=14, fontweight="bold",
                 color="white" if cm_feas[i, j] > cm_feas.max() / 2 else NAVY)
ax2.set_title("Model 2: Feasibility Confusion Matrix",
              fontweight="bold", color=NAVY, pad=12)
ax2.set_xlabel("Predicted", color=NAVY)
ax2.set_ylabel("Actual", color=NAVY)
ax2.text(0.05, 0.06, f"Accuracy: {acc_feas*100:.1f}%",
         transform=ax2.transAxes, fontsize=9,
         color="white", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN, alpha=0.9))

# ── Chart 3: Skill Level Confusion Matrix ────────────────────────────────────
ax3 = axes[1, 0]
ax3.set_facecolor("white")
cm_skill = confusion_matrix(y_test_skill, y_pred_skill)
ax3.imshow(cm_skill, cmap="Blues", aspect="auto")
ax3.set_xticks(range(len(skill_labels)))
ax3.set_yticks(range(len(skill_labels)))
ax3.set_xticklabels(skill_labels, color=NAVY, rotation=15)
ax3.set_yticklabels(skill_labels, color=NAVY)
for i in range(len(skill_labels)):
    for j in range(len(skill_labels)):
        ax3.text(j, i, str(cm_skill[i, j]),
                 ha="center", va="center",
                 fontsize=12, fontweight="bold",
                 color="white" if cm_skill[i, j] > cm_skill.max() / 2 else NAVY)
ax3.set_title("Model 3: Skill Level Confusion Matrix",
              fontweight="bold", color=NAVY, pad=12)
ax3.set_xlabel("Predicted", color=NAVY)
ax3.set_ylabel("Actual", color=NAVY)
ax3.text(0.05, 0.06, f"Accuracy: {acc_skill*100:.1f}%",
         transform=ax3.transAxes, fontsize=9,
         color="white", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.4", facecolor=GREEN, alpha=0.9))

# ── Chart 4: Model Accuracy Summary Bar Chart ────────────────────────────────
ax4 = axes[1, 1]
ax4.set_facecolor("white")
model_names = ["Cost Predictor\n(R² Score)", "Feasibility\nPredictor", "Complexity\nClassifier"]
model_scores = [r2 * 100, acc_feas * 100, acc_skill * 100]
bar_colors = [ACCENT, GREEN, YELLOW]
bars = ax4.bar(model_names, model_scores,
               color=bar_colors, edgecolor=NAVY,
               linewidth=0.8, width=0.5)
for bar, score in zip(bars, model_scores):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.5,
             f"{score:.1f}%",
             ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=NAVY)
ax4.set_ylim(0, 115)
ax4.set_title("All Models: Performance Summary",
              fontweight="bold", color=NAVY, pad=12)
ax4.set_ylabel("Score (%)", color=NAVY)
ax4.axhline(y=80, color=RED, linestyle="--",
            linewidth=1.5, alpha=0.7, label="80% threshold")
ax4.legend(fontsize=9)
ax4.spines[['top', 'right']].set_visible(False)
ax4.grid(True, axis="y", alpha=0.3)
ax4.tick_params(colors=NAVY)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight",
            facecolor="#f8fafc")
plt.close()

print("  Saved: model_evaluation.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  EVALUATION SUMMARY")
print("=" * 60)
print(f"  Model 1 — Cost Predictor        MAE = R{mae:.2f},  R² = {r2:.3f}")
print(f"  Model 2 — Feasibility Predictor  Accuracy = {acc_feas*100:.1f}%")
print(f"  Model 3 — Complexity Classifier  Accuracy = {acc_skill*100:.1f}%")
print("=" * 60)
print("\n  All charts saved to model_evaluation.png")
print("  Evaluation complete!")