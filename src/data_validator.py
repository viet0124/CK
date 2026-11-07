"""
Module nạp, chuyển đổi và kiểm tra dữ liệu bàn giao từ Ku em 1 (Bước 1).
Hỗ trợ:
1. Nạp đồ thị dị thể (HeteroGraph) từ `processed_graph.pt` và các cặp cạnh `train/val/test_pairs.pt`.
2. Tổng hợp đặc trưng nút `customer` (`x`, 60 chiều: 44 đặc trưng hành vi/tài khoản/thiết bị/IP/giao dịch
   + 16 đặc trưng vị trí cấu trúc lân cận) và chiếu các kênh quan hệ (`RELATED_TO`, `USES_DEVICE`,
   `USES_IP`, `TRANSACTS`) về đồ thị đồng nhất (`edge_index`) phục vụ GCN thuần và GAT thuần.
3. Kiểm tra thống kê dữ liệu, kiểm tra NaN/Inf và xác minh chống rò rỉ cạnh đích (Edge Leakage Check).
"""

import os
from typing import Dict, Optional, Tuple
import torch
import pandas as pd


DEFAULT_PROCESSED_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "processed",
    "results",
    "processed_data",
)

NEGATIVE_TYPE_NAMES = {
    0: "Positive (Đồng phạm cùng kịch bản - y=1)",
    1: "Type 1: AML-AML (Khác kịch bản - y=0)",
    2: "Type 2: AML-Clean (Tội phạm - Người sạch - y=0)",
    3: "Type 3: Suspected KYC (Nghi vấn giả - y=0)",
    4: "Type 4: Shared IP/Device (Người sạch chung mạng - y=0)",
    5: "Type 5: Clean-Clean (Ngẫu nhiên hoàn toàn - y=0)",
}

EDGE_CHANNEL_NAMES = {
    0: "RELATED_TO (KYC)",
    1: "SHARED_DEVICE",
    2: "SHARED_IP",
    3: "TRANSACTS (High-Value/Multi)",
}


def _bipartite_to_customer_edges(
    edge_idx: torch.Tensor, acc2cust: torch.Tensor
) -> torch.Tensor:
    """
    Chiếu liên kết 2 phía (account -> entity) thành cạnh đồng nhất (customer <-> customer)
    khi 2 khách hàng khác nhau cùng sử dụng chung 1 thực thể (Device hoặc IP).
    """
    cust = acc2cust[edge_idx[0]]
    ent = edge_idx[1]
    order = torch.argsort(ent)
    cust, ent = cust[order], ent[order]
    _, counts = torch.unique_consecutive(ent, return_counts=True)
    splits = torch.split(cust, counts.tolist())

    pairs = []
    for grp in splits:
        if grp.numel() >= 2:
            grp_u = grp.unique()
            if grp_u.numel() >= 2:
                u = grp_u.repeat_interleave(grp_u.numel())
                v = grp_u.repeat(grp_u.numel())
                mask = u != v
                pairs.append(torch.stack([u[mask], v[mask]], dim=0))
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat(pairs, dim=1).unique(dim=1)


def _compute_structural_pe(edge_index: torch.Tensor, num_nodes: int, dim: int = 16) -> torch.Tensor:
    """
    Tính vector nhúng vị trí cấu trúc mượt (Smooth Structural Neighborhood Encoding, 16 chiều)
    bằng 2 bước lan truyền chuẩn hóa bậc trên `edge_index` (không dùng bất kỳ nhãn y nào).
    Giúp GCN/GAT phân biệt hai nút thuộc cùng cụm giao dịch/thiết bị với hai nút ở hai kịch bản rời nhau.
    """
    gen = torch.Generator()
    gen.manual_seed(0)
    r = torch.randn((num_nodes, dim), generator=gen, dtype=torch.float32)
    r = r / r.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    loop_nodes = torch.arange(num_nodes, dtype=edge_index.dtype)
    ei = torch.cat([edge_index, torch.stack([loop_nodes, loop_nodes], dim=0)], dim=1)
    row, col = ei[0], ei[1]

    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.index_add_(0, col, torch.ones(col.shape[0], dtype=torch.float32))
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    for _ in range(2):
        out = torch.zeros_like(r)
        out.index_add_(0, col, r[row] * norm.unsqueeze(-1))
        r = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return r


def build_homogeneous_customer_graph(
    data_dir: str = DEFAULT_PROCESSED_DIR,
    include_covert_channels: bool = True,
    tx_amount_z_threshold: float = 1.2,
) -> Dict:
    """
    Nạp `processed_graph.pt` và chuyển thành đồ thị đồng nhất trên 50,000 nút `customer`
    để chạy GCN thuần và GAT thuần:
    - `x`: FloatTensor [50000, 60] đã chuẩn hóa, không NaN/Inf.
    - `edge_index`: LongTensor [2, E] gồm các cạnh truyền tin hai chiều.
    - `edge_channel`: LongTensor [E] lưu loại kênh cạnh (0: KYC, 1: Device, 2: IP, 3: Transacts).
    - `node_id_map`: Ánh xạ chỉ số nút -> Mã định danh hiển thị (`CUST_00000`).
    """
    graph_path = os.path.join(data_dir, "processed_graph.pt")
    raw_graph = torch.load(graph_path, weights_only=False)

    nf = raw_graph["node_features"]
    ei = raw_graph["edge_index"]
    ea = raw_graph["edge_attr"]

    x_cust = nf["customer"].float()
    num_customers = x_cust.shape[0]

    # Ánh xạ account -> customer thông qua quan hệ OWNS
    owns = ei[("customer", "OWNS", "account")]
    num_accounts = nf["account"].shape[0]
    acc2cust = torch.zeros(num_accounts, dtype=torch.long)
    acc2cust[owns[1]] = owns[0]

    # 1. Tổng hợp đặc trưng Account cho từng Customer (mean, max, count) -> 13 chiều
    x_acc = nf["account"].float()
    acc_sum = torch.zeros((num_customers, x_acc.shape[1]), dtype=torch.float32)
    acc_cnt = torch.zeros((num_customers, 1), dtype=torch.float32)
    acc_sum.index_add_(0, owns[0], x_acc[owns[1]])
    acc_cnt.index_add_(0, owns[0], torch.ones((owns.shape[1], 1), dtype=torch.float32))
    acc_mean = acc_sum / acc_cnt.clamp(min=1.0)

    acc_max = torch.full((num_customers, x_acc.shape[1]), -1e9, dtype=torch.float32)
    acc_max.scatter_reduce_(
        0,
        owns[0].unsqueeze(1).expand(-1, x_acc.shape[1]),
        x_acc[owns[1]],
        reduce="amax",
        include_self=True,
    )
    acc_max = torch.where(acc_max < -1e8, torch.zeros_like(acc_max), acc_max)

    # 2. Tổng hợp đặc trưng Device + USES_DEVICE edge_attr -> 9 chiều
    ad = ei[("account", "USES_DEVICE", "device")]
    c_ad = acc2cust[ad[0]]
    feat_ad = torch.cat([nf["device"][ad[1]].float(), ea[("account", "USES_DEVICE", "device")].float()], dim=1)
    dev_sum = torch.zeros((num_customers, feat_ad.shape[1]), dtype=torch.float32)
    dev_cnt = torch.zeros((num_customers, 1), dtype=torch.float32)
    dev_sum.index_add_(0, c_ad, feat_ad)
    dev_cnt.index_add_(0, c_ad, torch.ones((c_ad.shape[0], 1), dtype=torch.float32))
    dev_mean = dev_sum / dev_cnt.clamp(min=1.0)

    # 3. Tổng hợp đặc trưng IP + USES_IP edge_attr -> 7 chiều
    ai = ei[("account", "USES_IP", "ip")]
    c_ai = acc2cust[ai[0]]
    feat_ai = torch.cat([nf["ip"][ai[1]].float(), ea[("account", "USES_IP", "ip")].float()], dim=1)
    ip_sum = torch.zeros((num_customers, feat_ai.shape[1]), dtype=torch.float32)
    ip_cnt = torch.zeros((num_customers, 1), dtype=torch.float32)
    ip_sum.index_add_(0, c_ai, feat_ai)
    ip_cnt.index_add_(0, c_ai, torch.ones((c_ai.shape[0], 1), dtype=torch.float32))
    ip_mean = ip_sum / ip_cnt.clamp(min=1.0)

    # 4. Tổng hợp đặc trưng giao dịch TRANSACTS -> 5 chiều
    tx = ei[("account", "TRANSACTS", "account")]
    ea_tx = ea[("account", "TRANSACTS", "account")].float()
    c_tx_u = acc2cust[tx[0]]
    c_tx_v = acc2cust[tx[1]]
    tx_sum = torch.zeros((num_customers, ea_tx.shape[1]), dtype=torch.float32)
    tx_cnt = torch.zeros((num_customers, 1), dtype=torch.float32)
    tx_sum.index_add_(0, c_tx_u, ea_tx)
    tx_cnt.index_add_(0, c_tx_u, torch.ones((c_tx_u.shape[0], 1), dtype=torch.float32))
    tx_mean = tx_sum / tx_cnt.clamp(min=1.0)

    x_raw = torch.cat(
        [
            x_cust,
            acc_mean,
            acc_max,
            torch.log1p(acc_cnt),
            dev_mean,
            torch.log1p(dev_cnt),
            ip_mean,
            torch.log1p(ip_cnt),
            tx_mean,
            torch.log1p(tx_cnt),
        ],
        dim=1,
    )
    x_raw = torch.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
    feat_mean = x_raw.mean(dim=0, keepdim=True)
    feat_std = x_raw.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_norm = (x_raw - feat_mean) / feat_std

    # Xây dựng các kênh cạnh đồng nhất giữa các Customer (2 chiều u->v và v->u)
    e_rel = torch.cat(
        [ei[("customer", "RELATED_TO", "customer")], ei[("customer", "rev_RELATED_TO", "customer")]],
        dim=1,
    ).unique(dim=1)

    if include_covert_channels:
        e_dev = _bipartite_to_customer_edges(ad, acc2cust)
        e_ip = _bipartite_to_customer_edges(ai, acc2cust)

        mask_tx = (c_tx_u != c_tx_v) & ((ea_tx[:, 0] > tx_amount_z_threshold) | (ea_tx[:, 1] > 0.0))
        u_tx, v_tx = c_tx_u[mask_tx], c_tx_v[mask_tx]
        e_tx = torch.stack([torch.cat([u_tx, v_tx]), torch.cat([v_tx, u_tx])], dim=0).unique(dim=1)

        edge_list = [e_rel, e_dev, e_ip, e_tx]
        chan_list = [
            torch.full((e_rel.shape[1],), 0, dtype=torch.long),
            torch.full((e_dev.shape[1],), 1, dtype=torch.long),
            torch.full((e_ip.shape[1],), 2, dtype=torch.long),
            torch.full((e_tx.shape[1],), 3, dtype=torch.long),
        ]
        all_edges = torch.cat(edge_list, dim=1)
        all_chans = torch.cat(chan_list, dim=0)

        priority = torch.tensor([1, 0, 2, 3], dtype=torch.long)[all_chans]
        order = torch.argsort(priority)
        all_edges = all_edges[:, order]
        all_chans = all_chans[order]

        edge_hash = all_edges[0] * num_customers + all_edges[1]
        sorted_hash, sort_idx = torch.sort(edge_hash, stable=True)
        first_mask = torch.ones_like(sorted_hash, dtype=torch.bool)
        first_mask[1:] = sorted_hash[1:] != sorted_hash[:-1]
        keep_idx = sort_idx[first_mask]

        edge_index = all_edges[:, keep_idx]
        edge_channel = all_chans[keep_idx]
        channel_breakdown = {
            "RELATED_TO": int(e_rel.shape[1]),
            "SHARED_DEVICE": int(e_dev.shape[1]),
            "SHARED_IP": int(e_ip.shape[1]),
            "TRANSACTS_HIGH": int(e_tx.shape[1]),
        }
    else:
        edge_index = e_rel
        edge_channel = torch.zeros(e_rel.shape[1], dtype=torch.long)
        channel_breakdown = {
            "RELATED_TO": int(e_rel.shape[1]),
            "SHARED_DEVICE": 0,
            "SHARED_IP": 0,
            "TRANSACTS_HIGH": 0,
        }

    struct_pe = _compute_structural_pe(edge_index, num_customers, dim=16)
    x = torch.cat([x_norm, struct_pe], dim=1)

    node_id_map = {i: f"CUST_{i:05d}" for i in range(num_customers)}

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_channel": edge_channel,
        "rel_edge_index": e_rel,
        "channel_breakdown": channel_breakdown,
        "num_nodes": num_customers,
        "num_features": x.shape[1],
        "node_id_map": node_id_map,
        "raw_hetero_summary": {nt: tuple(t.shape) for nt, t in nf.items()},
    }


def load_split_pairs(data_dir: str = DEFAULT_PROCESSED_DIR) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Nạp các cặp cạnh huấn luyện / đánh giá / kiểm tra (`train_pairs.pt`, `val_pairs.pt`, `test_pairs.pt`).
    """
    splits = {}
    for split_name in ["train", "val", "test"]:
        path = os.path.join(data_dir, f"{split_name}_pairs.pt")
        splits[split_name] = torch.load(path, weights_only=False)
    return splits


def check_edge_leakage(
    graph_dict: Dict,
    splits: Dict[str, Dict[str, torch.Tensor]],
) -> Dict:
    """
    Kiểm tra tính tách bạch giữa quan hệ khai báo công khai (`RELATED_TO`) / nhãn đồng phạm (`labels.json`)
    và đồ thị đầu vào:
    1. Kiểm tra xem có bất kỳ cạnh đồng phạm dương (`y=1`) nào của `Train / Val / Test` xuất hiện trong
       quan hệ khai báo `RELATED_TO` (cả 2 chiều) hay không -> Phải bằng 0.
    2. Kiểm tra tính phân tách kịch bản (Scenario Split): Tập nút dương giữa `Train`, `Val`, `Test`
       không giao nhau (`overlap = 0`).
    """
    num_nodes = graph_dict["num_nodes"]
    rel_ei = graph_dict["rel_edge_index"]
    rel_hashes = torch.unique(rel_ei[0] * num_nodes + rel_ei[1])

    leak_counts = {}
    for s_name in ["train", "val", "test"]:
        pos = splits[s_name]["edge_label_index"][:, splits[s_name]["edge_label"] == 1]
        pos_bidir = torch.cat([pos, pos.flip(0)], dim=1)
        pos_h = pos_bidir[0] * num_nodes + pos_bidir[1]
        leak_counts[f"{s_name}_pos_in_RELATED_TO"] = int(torch.isin(pos_h, rel_hashes).sum().item())

    tr_nodes = set(splits["train"]["edge_label_index"][:, splits["train"]["edge_label"] == 1].flatten().tolist())
    va_nodes = set(splits["val"]["edge_label_index"][:, splits["val"]["edge_label"] == 1].flatten().tolist())
    te_nodes = set(splits["test"]["edge_label_index"][:, splits["test"]["edge_label"] == 1].flatten().tolist())

    return {
        **leak_counts,
        "train_val_pos_node_overlap": len(tr_nodes & va_nodes),
        "train_test_pos_node_overlap": len(tr_nodes & te_nodes),
        "val_test_pos_node_overlap": len(va_nodes & te_nodes),
        "leakage_free": all(v == 0 for v in leak_counts.values()),
    }


def inspect_graph_data(
    graph_dict: Dict, splits: Dict[str, Dict[str, torch.Tensor]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trả về 2 bảng DataFrame tổng hợp:
    1. Thống kê tổng quan đồ thị (`num_nodes`, `num_edges`, `num_features`, `NaN/Inf`, số cạnh từng kênh).
    2. Phân bổ mẫu dương (`y=1`) và 5 loại mẫu âm (`y=0`) trên từng tập `Train / Val / Test`.
    """
    x = graph_dict["x"]
    ei = graph_dict["edge_index"]
    cb = graph_dict["channel_breakdown"]

    graph_summary = pd.DataFrame(
        [
            {"Chi_so": "Số lượng nút khách hàng (num_nodes)", "Gia_tri": int(x.shape[0])},
            {"Chi_so": "Kích thước đặc trưng nút (num_features)", "Gia_tri": int(x.shape[1])},
            {"Chi_so": "Tổng số cạnh truyền tin sau hợp nhất (edge_index)", "Gia_tri": int(ei.shape[1])},
            {"Chi_so": "  ├─ Kênh RELATED_TO (KYC 2 chiều)", "Gia_tri": cb["RELATED_TO"]},
            {"Chi_so": "  ├─ Kênh SHARED_DEVICE (Dùng chung thiết bị)", "Gia_tri": cb["SHARED_DEVICE"]},
            {"Chi_so": "  ├─ Kênh SHARED_IP (Dùng chung IP)", "Gia_tri": cb["SHARED_IP"]},
            {"Chi_so": "  └─ Kênh TRANSACTS (Giao dịch lớn/nhiều lần)", "Gia_tri": cb["TRANSACTS_HIGH"]},
            {"Chi_so": "Số giá trị NaN trong x", "Gia_tri": int(torch.isnan(x).sum().item())},
            {"Chi_so": "Số giá trị Inf trong x", "Gia_tri": int(torch.isinf(x).sum().item())},
        ]
    )

    rows = []
    for split_name in ["train", "val", "test"]:
        p = splits[split_name]
        y = p["edge_label"]
        et = p["edge_type"]
        row = {
            "Split": split_name.upper(),
            "Total_Pairs": int(y.numel()),
            "Positive (y=1)": int((y == 1).sum().item()),
            "Negative (y=0)": int((y == 0).sum().item()),
            "Type1_AML_AML": int((et == 1).sum().item()),
            "Type2_AML_Clean": int((et == 2).sum().item()),
            "Type3_Suspected": int((et == 3).sum().item()),
            "Type4_Shared_IP_Dev": int((et == 4).sum().item()),
            "Type5_Clean_Clean": int((et == 5).sum().item()),
        }
        rows.append(row)

    split_summary = pd.DataFrame(rows)
    return graph_summary, split_summary
