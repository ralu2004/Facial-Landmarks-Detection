"""
analyze_results.py
Facial Landmarks Detection — evaluation analysis and plotting.

Usage:
    python analyze_results.py

Expects folder structure:
    Facial-Landmarks-Detection/
    ├── results/          ← CSVs and output PNGs go here
    └── analysis/         ← script lives here
    
    results_A1.csv, results_A2.csv, results_A2_upper_bound.csv,
    results_A3.csv, results_A4.csv
    results_A1_frontal.csv, results_A2_frontal.csv,
    results_A2_upper_bound_frontal.csv, results_A3_frontal.csv,
    results_A4_frontal.csv

Outputs:
    - Console: per-approach and per-subset tables
    - Saved PNGs: charts for the presentation
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['font.size'] = 11

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.dirname(os.path.abspath(__file__))  # analysis/ folder

# loadd csv files
ALL_FILES = {
    "A1 — HSV+YCbCr+Otsu": "results_A1.csv",
    "A2 — center seed":     "results_A2.csv",
    "A2 — GT seed":         "results_A2_upper_bound.csv",
    "A3 — VJ+classical":    "results_A3.csv",
    "A4 — VJ full":         "results_A4.csv",
}

FRONTAL_FILES = {
    "A1 — HSV+YCbCr+Otsu": "results_A1_frontal.csv",
    "A2 — center seed":     "results_A2_frontal.csv",
    "A2 — GT seed":         "results_A2_upper_bound_frontal.csv",
    "A3 — VJ+classical":    "results_A3_frontal.csv",
    "A4 — VJ full":         "results_A4_frontal.csv",
}

SHORT   = ["A1", "A2", "A2-GT", "A3", "A4"]
COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

POSE_LABELS    = {1: "Frontal", 2: "Left profile", 3: "Right profile", 4: "Upward", 5: "Downward"}
SMILE_LABELS   = {1: "Smiling",    2: "Not smiling"}
GLASSES_LABELS = {1: "Glasses",    2: "No glasses"}

def load_dfs(file_dict):
    dfs = {}
    for label, fname in file_dict.items():
        path = os.path.join(BASE, fname)
        df = pd.read_csv(path)
        for col in ["eye_nme", "mouth_nme", "combined_nme"]:
            df[col] = df[col].replace(-1, float("nan"))
        df["approach"] = label
        dfs[label] = df
    return dfs

dfs_all      = load_dfs(ALL_FILES)
dfs_frontal  = load_dfs(FRONTAL_FILES)

# summary statistics
def summarize(df, total_images=None):
    n         = total_images if total_images else len(df)
    detected  = df["eye_detected"].sum()
    mouth_det = df["mouth_detected"].sum()
    eye_nme   = df.loc[df["eye_detected"]   == 1, "eye_nme"].mean()
    mouth_nme = df.loc[df["mouth_detected"]  == 1, "mouth_nme"].mean()
    combined  = df.loc[df["eye_detected"]    == 1, "combined_nme"].mean()
    failures  = (df.loc[df["eye_detected"]   == 1, "combined_nme"] > 0.1).sum()
    fail_rate = failures / detected * 100 if detected > 0 else 0
    return {
        "N":               n,
        "Eye det.%":       f"{detected/n*100:.1f}",
        "Mouth det.%":     f"{mouth_det/n*100:.1f}",
        "Eye NME":         f"{eye_nme:.4f}"  if not pd.isna(eye_nme)   else "—",
        "Mouth NME":       f"{mouth_nme:.4f}" if not pd.isna(mouth_nme) else "—",
        "Combined NME":    f"{combined:.4f}"  if not pd.isna(combined)  else "—",
        "Failure%":        f"{fail_rate:.1f}",
    }

def get_numeric(dfs, col, detected_col="eye_detected"):
    vals = []
    for df in dfs.values():
        v = df.loc[df[detected_col] == 1, col].mean()
        vals.append(float(v) if not pd.isna(v) else 0)
    return vals

def get_det_rate(dfs):
    return [df["eye_detected"].mean() * 100 for df in dfs.values()]

# overall summary
print("\n" + "="*80)
print("OVERALL RESULTS — ALL POSES (2000 images)")
print("="*80)
rows = []
for label, df in dfs_all.items():
    s = summarize(df, 2000)
    s["Approach"] = label
    rows.append(s)
print(pd.DataFrame(rows).set_index("Approach").to_string())

# frontal-only summary
print("\n" + "="*80)
print("FRONTAL ONLY (pose=1, 136 images)")
print("="*80)
rows = []
for label, df in dfs_frontal.items():
    s = summarize(df, 136)
    s["Approach"] = label
    rows.append(s)
print(pd.DataFrame(rows).set_index("Approach").to_string())

# per-pose breakdown
print("\n" + "="*80)
print("PER-POSE BREAKDOWN — eye NME (2000 images)")
print("="*80)
pose_rows = []
for pose_id, pose_name in POSE_LABELS.items():
    for label, df in dfs_all.items():
        sub = df[df["pose"] == pose_id]
        if len(sub) == 0: continue
        detected = sub["eye_detected"].sum()
        eye_nme  = sub.loc[sub["eye_detected"] == 1, "eye_nme"].mean()
        pose_rows.append({
            "Pose":      pose_name,
            "Approach":  label,
            "N":         len(sub),
            "Det.%":     f"{detected/len(sub)*100:.1f}",
            "Eye NME":   f"{eye_nme:.4f}" if not pd.isna(eye_nme) else "—",
        })
pose_df = pd.DataFrame(pose_rows)
for pose_name, group in pose_df.groupby("Pose", sort=False):
    print(f"\n--- {pose_name} (N per approach shown) ---")
    print(group.drop(columns="Pose").set_index("Approach").to_string())

# glasses and smile breakdown
for attr, labels in [("glasses", GLASSES_LABELS), ("smile", SMILE_LABELS)]:
    print("\n" + "="*80)
    print(f"{attr.upper()} BREAKDOWN — eye NME")
    print("="*80)
    for val, name in labels.items():
        print(f"\n--- {name} ---")
        for label, df in dfs_all.items():
            sub = df[df[attr] == val]
            det = sub["eye_detected"].sum()
            nme = sub.loc[sub["eye_detected"] == 1, "eye_nme"].mean()
            det_r = det / len(sub) * 100 if len(sub) > 0 else 0
            nme_s = f"{nme:.4f}" if not pd.isna(nme) else "—"
            print(f"  {label:35s}  N={len(sub):4d}  det={det_r:.1f}%  eye_NME={nme_s}")

# PLOTS

# Plot 1: Detection rate: all vs frontal
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (dfs, title, n) in zip(axes, [
    (dfs_all,     "All poses (N=2000)", 2000),
    (dfs_frontal, "Frontal only (N=136)", 136),
]):
    rates = get_det_rate(dfs)
    bars = ax.bar(SHORT, rates, color=COLORS)
    ax.set_ylabel("Eye Detection Rate (%)")
    ax.set_title(title)
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
fig.suptitle("Eye Detection Rate by Approach", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_detection_rate.png"))
plt.close()
print("\nSaved: plot_detection_rate.png")

# Plot 2: Eye NME: all vs frontal
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, (dfs, title) in zip(axes, [
    (dfs_all,     "All poses (N=2000)"),
    (dfs_frontal, "Frontal only (N=136)"),
]):
    nmes = get_numeric(dfs, "eye_nme")
    bars = ax.bar(SHORT, nmes, color=COLORS)
    ax.set_ylabel("Eye NME")
    ax.set_title(title)
    ax.axhline(0.1, color="red", linestyle="--", linewidth=1)
    ax.set_ylim(0, max(nmes) * 1.2)
    for bar, val in zip(bars, nmes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Eye NME by Approach", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_eye_nme_all_vs_frontal.png"))
plt.close()
print("Saved: plot_eye_nme_all_vs_frontal.png")

# Plot 3: Eye / Mouth / Combined NME: all poses
eye_nmes      = get_numeric(dfs_all, "eye_nme")
mouth_nmes    = get_numeric(dfs_all, "mouth_nme", "mouth_detected")
combined_nmes = get_numeric(dfs_all, "combined_nme")

x = range(len(SHORT))
w = 0.25
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar([i - w for i in x], eye_nmes,      w, label="Eye NME",      color="#4C72B0")
ax.bar([i     for i in x], mouth_nmes,    w, label="Mouth NME",    color="#DD8452")
ax.bar([i + w for i in x], combined_nmes, w, label="Combined NME", color="#55A868")
ax.set_xticks(list(x))
ax.set_xticklabels(SHORT)
ax.set_ylabel("NME")
ax.set_title("NME by Landmark Type — All Poses (2000 images)")
ax.axhline(0.1, color="red", linestyle="--", linewidth=1, label="Failure threshold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_nme_breakdown.png"))
plt.close()
print("Saved: plot_nme_breakdown.png")

# Plot 4: Eye NME by pose (poses 2, 3, 4) 
poses_to_plot = [2, 3, 4]
pose_names    = [POSE_LABELS[p] for p in poses_to_plot]
x = range(len(poses_to_plot))
w = 1.0 / (len(dfs_all) + 1)

fig, ax = plt.subplots(figsize=(11, 5))
for idx, (label, df) in enumerate(dfs_all.items()):
    nmes = []
    for pose_id in poses_to_plot:
        sub = df[(df["pose"] == pose_id) & (df["eye_detected"] == 1)]
        nmes.append(sub["eye_nme"].mean() if len(sub) > 0 else 0)
    offset = (idx - len(dfs_all)/2) * w + w/2
    ax.bar([i + offset for i in x], nmes, w, label=SHORT[idx], color=COLORS[idx])

ax.set_xticks(list(x))
ax.set_xticklabels(pose_names)
ax.set_ylabel("Eye NME")
ax.set_title("Eye NME by Pose and Approach")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_nme_by_pose.png"))
plt.close()
print("Saved: plot_nme_by_pose.png")

# Plot 5: Glasses vs no glasses
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (glasses_id, glasses_name) in zip(axes, GLASSES_LABELS.items()):
    nmes = [df.loc[(df["glasses"] == glasses_id) & (df["eye_detected"] == 1), "eye_nme"].mean()
            for df in dfs_all.values()]
    bars = ax.bar(SHORT, nmes, color=COLORS)
    ax.set_title(glasses_name)
    ax.set_ylabel("Eye NME")
    ax.set_ylim(0, max(nmes) * 1.2)
    for bar, val in zip(bars, nmes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Eye NME: Glasses vs No Glasses")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_nme_glasses.png"))
plt.close()
print("Saved: plot_nme_glasses.png")

# Plot 6: Smile vs neutral
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, (smile_id, smile_name) in zip(axes, SMILE_LABELS.items()):
    nmes = [df.loc[(df["smile"] == smile_id) & (df["eye_detected"] == 1), "eye_nme"].mean()
            for df in dfs_all.values()]
    bars = ax.bar(SHORT, nmes, color=COLORS)
    ax.set_title(smile_name)
    ax.set_ylabel("Eye NME")
    ax.set_ylim(0, max(nmes) * 1.2)
    for bar, val in zip(bars, nmes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Eye NME: Smiling vs Not Smiling")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_nme_smile.png"))
plt.close()
print("Saved: plot_nme_smile.png")

# Plot 7: Dataset pose distribution
pose_counts = {POSE_LABELS[k]: v for k, v in
               {1: 136, 2: 1113, 3: 7445, 4: 1111, 5: 195}.items()}
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(list(pose_counts.keys()), list(pose_counts.values()),
              color=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"])
ax.set_ylabel("Number of images")
ax.set_title("MTFL Dataset — Pose Distribution (10,001 images)")
for bar, val in zip(bars, pose_counts.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(val), ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "plot_pose_distribution.png"))
plt.close()
print("Saved: plot_pose_distribution.png")

print("\nDone. All plots saved.")
