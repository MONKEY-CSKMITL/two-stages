"""
reporting.py — สร้างกราฟและตารางสรุปผลหลังเทรนเสร็จ (ไฟล์นี้เป็น "ห้องสมุด" ไม่ได้รันเอง)

แยกออกมาจาก train.py เพราะเป็นคนละหน้าที่กัน (train.py = เทรน, ไฟล์นี้ = รายงานผล)
ทำให้แก้รูปแบบกราฟ/ตารางได้โดยไม่ต้องแตะโค้ดเทรนเลย

หมายเหตุเรื่องภาษา: ข้อความในกราฟใช้ภาษาอังกฤษทั้งหมด เพราะ matplotlib บนเครื่อง
ทั่วไปมักไม่มีฟอนต์ไทยติดตั้งไว้ ทำให้ตัวอักษรไทยกลายเป็นสี่เหลี่ยมว่าง (tofu)
ส่วนข้อความที่พิมพ์ออก terminal ยังเป็นภาษาไทยตามปกติ

ผลลัพธ์ที่สร้าง:
    plots/loss_curve.png                   กราฟ train_loss vs val_loss ราย epoch
    plots/metric_curve.png                 กราฟ val_F1 กับ val_AUC ราย epoch
    plots/confusion_matrix.png             ตารางสับสนรวมทุกปล้อง
    plots/confusion_matrix_per_level.png   ตารางสับสนแยกราย 15 ปล้อง
    tables/metrics_overall.csv             precision/recall/f1 รวมทุกปล้อง
    tables/metrics_per_level.csv           precision/recall/f1 แยกรายปล้อง
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # ใช้โหมดที่ไม่ต้องเปิดหน้าต่าง (จำเป็นเวลารันบน server/ไม่มีจอ)
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# ชื่อปล้องเรียงตามลำดับกายวิภาค (index 0 = level_index 1 = T3)
LEVEL_NAMES = ["T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
               "T11", "T12", "L1", "L2", "L3", "L4", "L5"]


# ============================================================================
# ส่วนที่ 1: กราฟติดตามการเทรนราย epoch
# ============================================================================

def plot_loss_curve(history: pd.DataFrame, out_path: Path):
    """
    กราฟเส้น train_loss กับ val_loss ราย epoch

    วิธีอ่าน:
      - ทั้ง 2 เส้นลดลงพร้อมกัน = เรียนรู้ได้ดี
      - train ลดแต่ val เริ่มขึ้น = เริ่ม overfit (จำข้อมูลเทรนแทนที่จะเข้าใจ)
      - ทั้ง 2 เส้นนิ่งไม่ลด = เรียนรู้ไม่ได้ (อาจต้องปรับ learning rate)
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(history["epoch"], history["train_loss"], marker="o", label="train loss")
    ax.plot(history["epoch"], history["val_loss"], marker="s", label="val loss")

    # ทำเครื่องหมายจุดที่ val_loss ต่ำสุด (จุดที่โมเดลน่าจะดีที่สุดก่อนเริ่ม overfit)
    best_idx = history["val_loss"].idxmin()
    ax.axvline(history.loc[best_idx, "epoch"], color="gray", linestyle="--", linewidth=1)
    ax.annotate(f"lowest val loss\n(epoch {int(history.loc[best_idx, 'epoch'])})",
                xy=(history.loc[best_idx, "epoch"], history.loc[best_idx, "val_loss"]),
                xytext=(5, 15), textcoords="offset points", fontsize=8, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_metric_curve(history: pd.DataFrame, out_path: Path):
    """
    กราฟ val_F1 กับ val_AUC ราย epoch (แยกเป็น 2 แผงเพราะสเกลต่างกัน)

    วิธีอ่าน: F1 คือตัวที่ใช้เลือกโมเดลที่ดีที่สุด ดูว่าขึ้นถึงจุดสูงสุดตอน epoch ไหน
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # แผงซ้าย: macro F1
    axes[0].plot(history["epoch"], history["val_f1"], marker="o", color="#4C72B0")
    best_idx = history["val_f1"].idxmax()
    axes[0].axvline(history.loc[best_idx, "epoch"], color="gray", linestyle="--", linewidth=1)
    axes[0].annotate(f"best (epoch {int(history.loc[best_idx, 'epoch'])})",
                     xy=(history.loc[best_idx, "epoch"], history.loc[best_idx, "val_f1"]),
                     xytext=(5, -15), textcoords="offset points", fontsize=8, color="gray")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Macro F1")
    axes[0].set_title("Validation Macro F1 (model selection metric)")
    axes[0].grid(alpha=0.3)

    # แผงขวา: AUC
    axes[1].plot(history["epoch"], history["val_auc"], marker="s", color="#DD8452")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Validation AUC")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ============================================================================
# ส่วนที่ 2: ตารางสับสน (confusion matrix)
# ============================================================================

def _draw_confusion(ax, cm, class_names, title, show_labels=True, fontsize=9):
    """
    วาดตารางสับสน 1 แผงลงบน axes ที่ให้มา (ฟังก์ชันช่วย ใช้ภายในไฟล์นี้)

    แสดง 2 ตัวเลขในแต่ละช่อง: จำนวนดิบ และ % ของแถวนั้น (= recall ของคลาสนั้น)
    """
    # normalize ตามแถว = ดูว่า "ของจริงคลาสนี้ ถูกทายไปเป็นอะไรบ้าง กี่ %"
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_pct = cm / cm.sum(axis=1, keepdims=True)
        cm_pct = np.nan_to_num(cm_pct)   # แถวที่ไม่มีตัวอย่างเลย (หาร 0) ให้เป็น 0

    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    if show_labels:
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fontsize)
        ax.set_yticklabels(class_names, fontsize=fontsize)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # เขียนตัวเลขลงในแต่ละช่อง
    for i in range(n):
        for j in range(n):
            # เลือกสีตัวอักษรให้อ่านออกทั้งบนพื้นอ่อนและพื้นเข้ม
            color = "white" if cm_pct[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_pct[i, j]*100:.0f}%",
                    ha="center", va="center", color=color, fontsize=fontsize - 1)

    ax.set_title(title, fontsize=fontsize + 1)
    return im


def plot_confusion_matrix(targets, preds, class_names, out_path: Path):
    """
    ตารางสับสนรวมทุกปล้อง (แถว = คำตอบจริง, คอลัมน์ = ที่โมเดลทาย)

    วิธีอ่าน: แนวทแยงมุม (บนซ้าย→ล่างขวา) คือทายถูก ยิ่งเข้มยิ่งดี
              ช่องนอกแนวทแยง คือทายผิด บอกได้ว่าสับสนระหว่างคลาสไหนกับคลาสไหน
    """
    n = len(class_names)
    cm = confusion_matrix(targets, preds, labels=list(range(n)))

    fig, ax = plt.subplots(figsize=(1.4 * n + 2.5, 1.4 * n + 1.5))
    im = _draw_confusion(ax, cm, class_names, "Confusion Matrix (all levels)", fontsize=11)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, label="proportion of true class (row-normalized)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_confusion_matrix_per_level(targets, preds, levels, class_names, out_path: Path):
    """
    ตารางสับสนแยกราย 15 ปล้อง วางเป็นตาราง 3 แถว x 5 คอลัมน์

    levels = level_index ของแต่ละตัวอย่าง (1-15)

    ประโยชน์: ดูว่าโมเดลทำได้ดี/แย่เฉพาะบางปล้องไหม เช่น ปล้องช่วง T11-L1
    ที่มีการหักเยอะ อาจทำได้ดีกว่าปล้องบนที่แทบไม่มีตัวอย่างการหักเลย
    """
    n_cls = len(class_names)
    fig, axes = plt.subplots(3, 5, figsize=(19, 12))
    axes = axes.flatten()

    for i, lv_name in enumerate(LEVEL_NAMES):
        ax = axes[i]
        lv_index = i + 1                    # LEVEL_NAMES index 0 = level_index 1
        mask = levels == lv_index
        n_samples = int(mask.sum())

        if n_samples == 0:
            # ปล้องนี้ไม่มีตัวอย่างในชุดที่วัดผลเลย
            ax.text(0.5, 0.5, f"{lv_name}\n(no samples)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        cm = confusion_matrix(targets[mask], preds[mask], labels=list(range(n_cls)))
        # แสดง label เฉพาะแผงริมซ้าย/ล่าง เพื่อไม่ให้รก
        _draw_confusion(ax, cm, class_names, f"{lv_name}  (n={n_samples})",
                        show_labels=True, fontsize=7)

    fig.suptitle("Confusion Matrix per Vertebral Level", fontsize=15, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ============================================================================
# ส่วนที่ 3: ตารางตัวเลข (CSV)
# ============================================================================

def save_metrics_overall(targets, preds, class_names, out_path: Path) -> pd.DataFrame:
    """
    ตาราง precision/recall/f1/support แยกตามคลาส รวมทุกปล้อง

    zero_division=0 = ถ้าคลาสไหนไม่มีตัวอย่างเลย หรือโมเดลไม่เคยทายคลาสนั้นเลย
    ให้ใส่ 0 แทนที่จะ error (เกิดได้จริงกับคลาสที่มีตัวอย่างน้อยมาก)
    """
    n = len(class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(n)), zero_division=0)

    df = pd.DataFrame({
        "class": class_names,
        "precision": precision.round(4),
        "recall": recall.round(4),
        "f1_score": f1.round(4),
        "support": support,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def save_metrics_per_level(targets, preds, levels, class_names, out_path: Path) -> pd.DataFrame:
    """
    ตาราง precision/recall/f1/support แยกทั้งรายปล้องและรายคลาส
    (15 ปล้อง x จำนวนคลาส = 60 แถวสำหรับโจทย์ 4 คลาส)

    ข้อควรระวังในการตีความ: หลายช่องจะมี support น้อยมาก (เช่น ปล้อง T3 อาจไม่มี
    ตัวอย่างที่หักเลย) ตัวเลข precision/recall ของช่องที่ support น้อยจะผันผวนสูง
    ไม่ควรตีความจริงจัง — ดูคอลัมน์ support ประกอบเสมอ
    """
    n = len(class_names)
    rows = []

    for i, lv_name in enumerate(LEVEL_NAMES):
        lv_index = i + 1
        mask = levels == lv_index
        if mask.sum() == 0:
            continue   # ปล้องนี้ไม่มีตัวอย่างในชุดนี้เลย ข้ามไป

        precision, recall, f1, support = precision_recall_fscore_support(
            targets[mask], preds[mask], labels=list(range(n)), zero_division=0)

        for c in range(n):
            rows.append({
                "level": lv_name,
                "level_index": lv_index,
                "class": class_names[c],
                "precision": round(float(precision[c]), 4),
                "recall": round(float(recall[c]), 4),
                "f1_score": round(float(f1[c]), 4),
                "support": int(support[c]),
            })

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def save_history(history: pd.DataFrame, out_path: Path):
    """บันทึกค่าราย epoch ลง CSV (ไว้ทำกราฟใหม่เองทีหลัง หรือเทียบหลาย run)"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(out_path, index=False)


# ============================================================================
# ส่วนที่ 4: ฟังก์ชันรวม เรียกทีเดียวได้ผลครบ
# ============================================================================

def generate_all_reports(history: pd.DataFrame, targets, preds, levels,
                         class_names, out_dir: Path):
    """
    สร้างกราฟและตารางทั้งหมดในครั้งเดียว — train.py เรียกแค่ฟังก์ชันนี้ฟังก์ชันเดียว

    history = ตารางค่าราย epoch (คอลัมน์: epoch, train_loss, val_loss, val_f1, val_auc)
    targets/preds/levels = ผลบนชุด test
    """
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # กราฟติดตามการเทรน
    plot_loss_curve(history, plots_dir / "loss_curve.png")
    plot_metric_curve(history, plots_dir / "metric_curve.png")
    save_history(history, out_dir / "history.csv")

    # ตารางสับสน
    plot_confusion_matrix(targets, preds, class_names, plots_dir / "confusion_matrix.png")
    plot_confusion_matrix_per_level(targets, preds, levels, class_names,
                                    plots_dir / "confusion_matrix_per_level.png")

    # ตารางตัวเลข
    df_overall = save_metrics_overall(targets, preds, class_names,
                                      tables_dir / "metrics_overall.csv")
    df_per_level = save_metrics_per_level(targets, preds, levels, class_names,
                                          tables_dir / "metrics_per_level.csv")

    return df_overall, df_per_level