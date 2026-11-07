"""
Module trực quan hóa phục vụ Báo cáo & Case Study (Bước 5, Bước 6 & Bước 7):
1. `plot_training_curves`: Vẽ đường cong Train/Val Loss và Validation AP theo epoch cho GCN & GAT.
2. `plot_scenario_comparison`: Vẽ biểu đồ cột so sánh GCN và GAT (Mean ± Std) qua các kịch bản.
3. `find_case_study_examples`: Tự động chọn từ tập Test:
   - 1 trường hợp Dự đoán Đúng điển hình (True Positive: cặp đồng phạm thực sự phát hiện qua kênh ngầm Device/IP/Transacts).
   - 1 trường hợp Dự đoán Sai điển hình (False Positive trên Hard Negative Type 4/Type 1 hoặc False Negative).
4. `plot_case_study_comparison`: Vẽ vùng mạng nhỏ quanh cặp nút mục tiêu theo đúng quy ước:
   - Cạnh đầu vào đã quan sát (`edge_index`): Nét liền (`solid`).
   - Liên kết mô hình đề xuất: Nét đứt (`dashed`).
   - Cạnh đúng bị che trong thực nghiệm (`ground-truth test edge`): Chú thích đối chiếu riêng.
   - Bảng Điểm dự đoán (`Score`) và Thứ hạng (`Rank / 51`) hiển thị ngay cạnh hình.
5. `plot_gat_attention_case_study`: Trực quan hóa trọng số Attention của GAT trên vùng mạng con.
"""

import os
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
import torch


CHANNEL_LABELS = {
    0: "RELATED_TO (KYC)",
    1: "SHARED_DEVICE",
    2: "SHARED_IP",
    3: "TRANSACTS",
}

CHANNEL_COLORS = {
    0: "#2b5c8f",  # Xanh dương đậm (KYC)
    1: "#d95f02",  # Cam đậm (Shared Device)
    2: "#7570b3",  # Tím (Shared IP)
    3: "#66a61e",  # Xanh lá (Transacts)
}


def plot_training_curves(
    histories: Dict[str, List[Dict[str, float]]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ Loss (Train / Val) và Validation AP theo Epoch cho GCN và GAT.
    `histories` có dạng `{"GCN": history_list, "GAT": history_list}`.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=140)
    colors = {"GCN": "#1f77b4", "GAT": "#d62728"}

    for model_name, hist in histories.items():
        epochs = [h["epoch"] for h in hist]
        tr_loss = [h["train_loss"] for h in hist]
        va_loss = [h["val_loss"] for h in hist]
        va_ap = [h["val_ap"] for h in hist]
        va_auc = [h["val_roc_auc"] for h in hist]
        c = colors.get(model_name, "#333333")

        axes[0].plot(epochs, tr_loss, label=f"{model_name} - Train Loss", color=c, linestyle="-", linewidth=2)
        axes[0].plot(epochs, va_loss, label=f"{model_name} - Val Loss", color=c, linestyle="--", linewidth=1.8, alpha=0.85)

        best_idx = int(np.argmax(va_ap))
        axes[1].plot(epochs, va_ap, label=f"{model_name} - Val AP (Best={va_ap[best_idx]:.4f})", color=c, linestyle="-", linewidth=2.2)
        axes[1].plot(epochs, va_auc, label=f"{model_name} - Val ROC-AUC", color=c, linestyle=":", linewidth=1.6, alpha=0.8)
        axes[1].scatter([epochs[best_idx]], [va_ap[best_idx]], color=c, s=70, zorder=5, edgecolors="black")

    axes[0].set_title("A. Quá trình Hội tụ Hàm Loss (BCEWithLogitsLoss)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE Loss")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(frameon=True, fontsize=9)

    axes[1].set_title("B. Đường cong Validation AP & ROC-AUC (Chọn Checkpoint)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(frameon=True, fontsize=9)

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=160)
    return fig


def plot_scenario_comparison(
    summary_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ biểu đồ cột có thanh sai số (Mean ± Std) so sánh GCN và GAT trên các chỉ số
    ROC-AUC, AP, F1, Hits@10 qua các kịch bản thực nghiệm.
    """
    metrics = ["ROC-AUC", "AP", "F1", "Hits@10"]
    scenarios = summary_df["Scenario"].unique().tolist()
    x = np.arange(len(scenarios))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=140)
    axes = axes.flatten()

    for idx, m in enumerate(metrics):
        ax = axes[idx]
        gcn_rows = summary_df[summary_df["Model"] == "GCN"].set_index("Scenario").reindex(scenarios)
        gat_rows = summary_df[summary_df["Model"] == "GAT"].set_index("Scenario").reindex(scenarios)

        gcn_mean = gcn_rows[f"{m}_mean"].values
        gcn_std = gcn_rows[f"{m}_std"].values
        gat_mean = gat_rows[f"{m}_mean"].values
        gat_std = gat_rows[f"{m}_std"].values

        bars1 = ax.bar(x - width / 2, gcn_mean, width, yerr=gcn_std, capsize=4, label="GCN thuần", color="#2b5c8f", alpha=0.9)
        bars2 = ax.bar(x + width / 2, gat_mean, width, yerr=gat_std, capsize=4, label="GAT thuần", color="#d95f02", alpha=0.9)

        ax.set_title(f"So sánh {m} theo Kịch bản (Mean ± Std, 3 Seeds)", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("Scenario ", "S").replace(" (All 5 Neg Types)", "") for s in scenarios], rotation=20, ha="right", fontsize=8.5)
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8.5, loc="lower right")

        for b in list(bars1) + list(bars2):
            h = b.get_height()
            if not np.isnan(h):
                ax.annotate(
                    f"{h:.2f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=160)
    return fig


def _compute_pair_rank(score: float, neg_scores: np.ndarray, pool_size: int = 50, seed: int = 42, pair_idx: int = 0) -> Tuple[int, int]:
    """Tính thứ hạng (Rank) của 1 cặp mục tiêu so với nhóm 50 ứng viên âm cố định."""
    rng = np.random.RandomState(seed + pair_idx)
    n_neg = min(pool_size, len(neg_scores))
    sampled = rng.choice(len(neg_scores), size=n_neg, replace=False)
    cand = neg_scores[sampled]
    rank = 1 + int(np.sum(cand >= score))
    return rank, n_neg + 1


def select_case_study_pairs(
    edge_index: torch.Tensor,
    edge_channel: torch.Tensor,
    test_eli: torch.Tensor,
    test_y: np.ndarray,
    test_et: np.ndarray,
    gcn_scores: np.ndarray,
    gat_scores: np.ndarray,
    threshold: float,
    num_nodes: int = 50000,
) -> Dict[str, Dict]:
    """
    Chọn tự động từ tập kiểm tra (Test set):
    1. `correct_case`: Một cặp đồng phạm thật (`y=1`) có liên kết kênh ngầm (`SHARED_DEVICE` / `SHARED_IP` / `TRANSACTS`)
       hoặc lân cận chung, được cả GAT/GCN dự đoán đúng với điểm số cao và nằm trong Top-10 (`Hits@10`).
    2. `incorrect_case`: Một trường hợp mô hình dự đoán sai điển hình — ưu tiên mẫu âm khó (`Type 4: Shared IP/Device`
       giữa 2 người sạch hoặc `Type 1: AML-AML khác kịch bản`) bị mô hình chấm điểm vượt ngưỡng (False Positive),
       hoặc một cặp đồng phạm bị bỏ sót (False Negative) do cô lập trên đồ thị.
    """
    ei_np = edge_index.cpu().numpy()
    ec_np = edge_channel.cpu().numpy()
    eli_np = test_eli.cpu().numpy()

    # Lập bảng tra cứu cạnh trực tiếp (u, v) -> channel
    edge_map = {}
    adj = {}
    for idx in range(ei_np.shape[1]):
        u, v, c = int(ei_np[0, idx]), int(ei_np[1, idx]), int(ec_np[idx])
        edge_map[(u, v)] = c
        adj.setdefault(u, set()).add(v)

    neg_scores_gat = gat_scores[test_y == 0]
    neg_scores_gcn = gcn_scores[test_y == 0]

    # 1. Tìm ca dự đoán Đúng (True Positive) có cấu trúc đồ thị trực quan
    best_tp_idx = None
    best_tp_score = -1.0
    for i in range(len(test_y)):
        if test_y[i] != 1.0:
            continue
        u, v = int(eli_np[0, i]), int(eli_np[1, i])
        nbr_u = adj.get(u, set())
        nbr_v = adj.get(v, set())
        common = nbr_u & nbr_v
        has_direct = (u, v) in edge_map
        if (has_direct or len(common) >= 1) and len(nbr_u) + len(nbr_v) >= 3:
            score = float(gat_scores[i])
            if score > best_tp_score:
                best_tp_score = score
                best_tp_idx = i

    if best_tp_idx is None:
        pos_indices = np.where(test_y == 1.0)[0]
        best_tp_idx = int(pos_indices[np.argmax(gat_scores[pos_indices])])

    # 2. Tìm ca dự đoán Sai (False Positive trên Hard Negative Type 4 / Type 1 hoặc False Negative)
    best_err_idx = None
    best_err_margin = -1.0
    for i in range(len(test_y)):
        u, v = int(eli_np[0, i]), int(eli_np[1, i])
        nbr_u = adj.get(u, set())
        nbr_v = adj.get(v, set())
        if test_y[i] == 0.0 and gat_scores[i] >= threshold and len(nbr_u) + len(nbr_v) >= 2:
            # Ưu tiên Type 4 (Shared IP/Device) hoặc Type 1 (AML-AML) bị chấm điểm cao
            bonus = 0.25 if test_et[i] in (4, 1) else 0.0
            margin = float(gat_scores[i] - threshold) + bonus
            if margin > best_err_margin:
                best_err_margin = margin
                best_err_idx = i

    if best_err_idx is None:
        # Fallback: False Negative (y=1 nhưng điểm dự đoán thấp nhất dưới ngưỡng)
        fn_indices = np.where((test_y == 1.0) & (gat_scores < threshold))[0]
        if len(fn_indices) > 0:
            best_err_idx = int(fn_indices[np.argmin(gat_scores[fn_indices])])
        else:
            neg_indices = np.where(test_y == 0.0)[0]
            best_err_idx = int(neg_indices[np.argmax(gat_scores[neg_indices])])

    cases = {}
    for key, idx in [("correct_case", best_tp_idx), ("incorrect_case", best_err_idx)]:
        u, v = int(eli_np[0, idx]), int(eli_np[1, idx])
        gat_rk, pool_n = _compute_pair_rank(float(gat_scores[idx]), neg_scores_gat, pair_idx=idx)
        gcn_rk, _ = _compute_pair_rank(float(gcn_scores[idx]), neg_scores_gcn, pair_idx=idx)
        cases[key] = {
            "pair_idx": int(idx),
            "u": u,
            "v": v,
            "u_id": f"CUST_{u:05d}",
            "v_id": f"CUST_{v:05d}",
            "y_true": int(test_y[idx]),
            "edge_type": int(test_et[idx]),
            "gcn_score": float(gcn_scores[idx]),
            "gat_score": float(gat_scores[idx]),
            "threshold": float(threshold),
            "gcn_pred": int(gcn_scores[idx] >= threshold),
            "gat_pred": int(gat_scores[idx] >= threshold),
            "gcn_rank": f"{gcn_rk}/{pool_n}",
            "gat_rank": f"{gat_rk}/{pool_n}",
        }
    return cases


def extract_local_subgraph(
    u: int,
    v: int,
    edge_index: torch.Tensor,
    edge_channel: torch.Tensor,
    max_neighbors_per_endpoint: int = 6,
) -> Tuple[List[int], List[Tuple[int, int, int]]]:
    """
    Trích xuất vùng mạng nhỏ (ego-subgraph 1..2 hop) xung quanh cặp nút `(u, v)`.
    Ưu tiên giữ các nút trung gian chung (common neighbors) và các cạnh kênh ngầm hiếm (`Device`, `IP`, `KYC`).
    """
    ei_np = edge_index.cpu().numpy()
    ec_np = edge_channel.cpu().numpy()

    mask_u = ei_np[0] == u
    mask_v = ei_np[0] == v

    nbrs_u = [(int(ei_np[1, i]), int(ec_np[i])) for i in np.where(mask_u)[0] if int(ei_np[1, i]) != u]
    nbrs_v = [(int(ei_np[1, i]), int(ec_np[i])) for i in np.where(mask_v)[0] if int(ei_np[1, i]) != v]

    set_u = {n for n, _ in nbrs_u}
    set_v = {n for n, _ in nbrs_v}
    common = list(set_u & set_v)

    # Sắp xếp ưu tiên kênh hiếm (Device=1, IP=2, KYC=0 trước Transacts=3)
    nbrs_u.sort(key=lambda x: (0 if x[0] in common else 1, x[1]))
    nbrs_v.sort(key=lambda x: (0 if x[0] in common else 1, x[1]))

    selected_nodes = {u, v}
    for n in common[:max_neighbors_per_endpoint]:
        selected_nodes.add(n)
    for n, _ in nbrs_u[:max_neighbors_per_endpoint]:
        selected_nodes.add(n)
    for n, _ in nbrs_v[:max_neighbors_per_endpoint]:
        selected_nodes.add(n)

    node_list = sorted(selected_nodes)
    node_set = set(node_list)

    # Lấy toàn bộ cạnh quan sát đầu vào giữa các nút trong vùng mạng con
    sub_edges = []
    seen_undir = set()
    for idx in range(ei_np.shape[1]):
        a, b = int(ei_np[0, idx]), int(ei_np[1, idx])
        if a in node_set and b in node_set and a != b:
            pair = (min(a, b), max(a, b))
            if pair not in seen_undir:
                seen_undir.add(pair)
                sub_edges.append((a, b, int(ec_np[idx])))

    return node_list, sub_edges


def plot_case_study_figure(
    cases: Dict[str, Dict],
    edge_index: torch.Tensor,
    edge_channel: torch.Tensor,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Vẽ hình Case Study (Bước 6) gồm 2 khung:
    - Trái: Trường hợp Dự đoán Đúng (True Positive).
    - Phải: Trường hợp Dự đoán Sai (False Positive / False Negative).
    Quy ước bắt buộc:
    - Cạnh đầu vào đã quan sát: Nét liền (`solid`).
    - Liên kết mô hình đề xuất: Nét đứt (`dashed`).
    - Cạnh đúng bị che trong thực nghiệm (`ground-truth accomplice`): Đường viền vàng/xanh lá nổi bật + chú thích riêng.
    - Điểm dự đoán & Thứ hạng hiển thị ngay trong khung bảng dưới mỗi đồ thị.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=150)

    panels = [
        ("correct_case", axes[0], "A. Case Study 1 — Trường hợp Dự đoán ĐÚNG (True Positive)"),
        ("incorrect_case", axes[1], "B. Case Study 2 — Trường hợp Dự đoán SAI (Error Analysis)"),
    ]

    for key, ax, title in panels:
        info = cases[key]
        u, v = info["u"], info["v"]
        nodes, obs_edges = extract_local_subgraph(u, v, edge_index, edge_channel, max_neighbors_per_endpoint=5)

        G = nx.Graph()
        for n in nodes:
            G.add_node(n)
        for a, b, c in obs_edges:
            G.add_edge(a, b, channel=c)

        # Cố định vị trí hai nút mục tiêu u, v ở trung tâm trái/phải để dễ quan sát
        pos = nx.spring_layout(G, seed=42, k=1.1)
        pos[u] = np.array([-0.45, 0.0])
        pos[v] = np.array([0.45, 0.0])
        other_nodes = [n for n in nodes if n not in (u, v)]
        angles = np.linspace(0, 2 * np.pi, max(len(other_nodes), 1), endpoint=False)
        for idx_n, n in enumerate(other_nodes):
            rad = 0.85 if (G.has_edge(u, n) and G.has_edge(v, n)) else 1.15
            pos[n] = np.array([rad * np.cos(angles[idx_n]), rad * np.sin(angles[idx_n])])

        # 1. Vẽ các cạnh đầu vào đã quan sát (NÉT LIỀN)
        for a, b, c in obs_edges:
            color = CHANNEL_COLORS.get(c, "#555555")
            ax.plot(
                [pos[a][0], pos[b][0]],
                [pos[a][1], pos[b][1]],
                linestyle="-",
                linewidth=2.0,
                color=color,
                alpha=0.8,
                zorder=2,
            )

        # 2. Nếu là cạnh đồng phạm đúng bị ẩn trong thực nghiệm (y_true == 1): vẽ lớp nền đối chiếu riêng
        if info["y_true"] == 1:
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                linestyle="-",
                linewidth=8.5,
                color="#ffd92f",
                alpha=0.45,
                zorder=3,
            )

        # 3. Vẽ liên kết mô hình đề xuất (NÉT ĐỨT) giữa (u, v)
        if info["gat_pred"] == 1:
            pred_color = "#d62728" if info["y_true"] == 1 else "#e7298a"
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                linestyle="--",
                linewidth=2.8,
                color=pred_color,
                zorder=4,
            )

        # 4. Vẽ các nút
        for n in nodes:
            if n in (u, v):
                node_color = "#d62728" if info["y_true"] == 1 else "#ff7f0e"
                size = 850
                label = f"CUST_{n:05d}\n(Target)"
            else:
                is_common = G.has_edge(u, n) and G.has_edge(v, n)
                node_color = "#66c2a5" if is_common else "#b3cde3"
                size = 520 if is_common else 380
                label = f"CUST_{n:05d}"

            ax.scatter(
                [pos[n][0]],
                [pos[n][1]],
                s=size,
                color=node_color,
                edgecolors="#222222",
                linewidths=1.5,
                zorder=5,
            )
            ax.text(
                pos[n][0],
                pos[n][1] + (0.16 if n in (u, v) else 0.13),
                label,
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight="bold" if n in (u, v) else "normal",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                zorder=6,
            )

        # Hộp thông tin Điểm dự đoán & Thứ hạng
        gt_str = "Đồng phạm thật (y=1, bị che)" if info["y_true"] == 1 else f"Mẫu âm (y=0, Type {info['edge_type']})"
        box_text = (
            f"Cặp kiểm tra: ({info['u_id']}, {info['v_id']})\n"
            f"Nhãn thực nghiệm: {gt_str}\n"
            f"GAT Score: {info['gat_score']:.4f} | Rank: {info['gat_rank']} | Pred: y={info['gat_pred']}\n"
            f"GCN Score: {info['gcn_score']:.4f} | Rank: {info['gcn_rank']} | Pred: y={info['gcn_pred']}\n"
            f"Ngưỡng F1 cố định (Val): {info['threshold']:.4f}"
        )
        ax.text(
            0.02,
            0.02,
            box_text,
            transform=ax.transAxes,
            fontsize=8.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f8f9fa", edgecolor="#444444", alpha=0.95),
            zorder=7,
        )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.55, 1.45)
        ax.axis("off")

    legend_elements = [
        Line2D([0], [0], color="#2b5c8f", lw=2.2, linestyle="-", label="Cạnh đầu vào đã quan sát: RELATED_TO (Nét liền)"),
        Line2D([0], [0], color="#d95f02", lw=2.2, linestyle="-", label="Cạnh đầu vào đã quan sát: SHARED_DEVICE (Nét liền)"),
        Line2D([0], [0], color="#7570b3", lw=2.2, linestyle="-", label="Cạnh đầu vào đã quan sát: SHARED_IP (Nét liền)"),
        Line2D([0], [0], color="#66a61e", lw=2.2, linestyle="-", label="Cạnh đầu vào đã quan sát: TRANSACTS (Nét liền)"),
        Line2D([0], [0], color="#d62728", lw=2.8, linestyle="--", label="Liên kết mô hình đề xuất (Nét đứt)"),
        Line2D([0], [0], color="#ffd92f", lw=7.0, linestyle="-", alpha=0.6, label="Cạnh đồng phạm đúng bị che trong thực nghiệm (Đối chiếu)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=8.8,
        frameon=True,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=160)
    return fig


def plot_gat_attention_subgraph(
    case_info: Dict,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_channel: torch.Tensor,
    gat_model: torch.nn.Module,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Phần bổ sung Bước 6: Trích xuất và minh họa trọng số Attention (trung bình qua các heads)
    của mô hình GAT trên vùng mạng con xung quanh cặp nút Case Study.
    """
    u, v = case_info["u"], case_info["v"]
    nodes, obs_edges = extract_local_subgraph(u, v, edge_index, edge_channel, max_neighbors_per_endpoint=5)
    node_set = set(nodes)

    dev = next(gat_model.parameters()).device
    _, (ei_1, alpha_1), (ei_2, alpha_2) = gat_model.encoder.encode_with_attention(
        x.to(dev), edge_index.to(dev)
    )
    ei_np = ei_2.cpu().numpy()
    alpha_mean = alpha_2.mean(dim=-1).cpu().numpy()

    attn_lookup = {}
    for idx in range(ei_np.shape[1]):
        a, b = int(ei_np[0, idx]), int(ei_np[1, idx])
        if a in node_set and b in node_set and a != b:
            pair = (min(a, b), max(a, b))
            attn_lookup[pair] = max(attn_lookup.get(pair, 0.0), float(alpha_mean[idx]))

    fig, ax = plt.subplots(figsize=(8.5, 6.2), dpi=150)
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for a, b, c in obs_edges:
        G.add_edge(a, b, channel=c)

    pos = {}
    pos[u] = np.array([-0.45, 0.0])
    pos[v] = np.array([0.45, 0.0])
    other_nodes = [n for n in nodes if n not in (u, v)]
    angles = np.linspace(0, 2 * np.pi, max(len(other_nodes), 1), endpoint=False)
    for idx_n, n in enumerate(other_nodes):
        rad = 0.85 if (G.has_edge(u, n) and G.has_edge(v, n)) else 1.15
        pos[n] = np.array([rad * np.cos(angles[idx_n]), rad * np.sin(angles[idx_n])])

    for a, b, c in obs_edges:
        pair = (min(a, b), max(a, b))
        w = attn_lookup.get(pair, 0.05)
        lw = 1.2 + 12.0 * min(w, 0.5)
        ax.plot(
            [pos[a][0], pos[b][0]],
            [pos[a][1], pos[b][1]],
            linestyle="-",
            linewidth=lw,
            color="#d95f02",
            alpha=0.75,
            zorder=2,
        )
        mid_x, mid_y = (pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2
        ax.text(
            mid_x,
            mid_y,
            f"α={w:.2f}",
            fontsize=7.5,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="#fff7bc", ec="#d95f02", alpha=0.9),
            zorder=4,
        )

    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], linestyle="--", linewidth=2.6, color="#d62728", zorder=3)

    for n in nodes:
        node_color = "#d62728" if n in (u, v) else "#66c2a5"
        size = 820 if n in (u, v) else 460
        ax.scatter([pos[n][0]], [pos[n][1]], s=size, color=node_color, edgecolors="#222222", linewidths=1.5, zorder=5)
        ax.text(
            pos[n][0],
            pos[n][1] + 0.14,
            f"CUST_{n:05d}",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold" if n in (u, v) else "normal",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
            zorder=6,
        )

    ax.set_title(
        f"Trọng số Attention của GAT quanh cặp ({case_info['u_id']}, {case_info['v_id']}) — Score={case_info['gat_score']:.4f}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.45, 1.45)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=160)
    return fig
