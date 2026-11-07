"""
Bộ tính Metrics dùng chung cho cả nhóm (Bước 3).
Đầu vào hoàn toàn độc lập với mô hình: chỉ nhận nhãn thật (`y_true`) và điểm dự đoán (`y_score`).
Các thành viên khác trong nhóm (ví dụ mô hình HeteroGNN / GraphSAGE / RGCN) có thể gọi trực tiếp
`evaluate_predictions` hoặc `evaluate_from_files` để xuất kết quả trên cùng một thước đo.

Các chỉ số bao gồm:
1. ROC-AUC: Tính từ điểm dự đoán liên tục.
2. AP (Average Precision): Tính từ điểm dự đoán liên tục.
3. F1-Score: Quét tìm ngưỡng tối ưu trên tập Validation, cố định ngưỡng đó khi chấm tập Test.
4. Hits@K (mặc định K=10):
   - Quy ước chuẩn thống nhất cho toàn nhóm: Mỗi cạnh dương (y=1) được xếp hạng cùng một nhóm
     gồm M=50 mẫu âm ứng viên cố định (sinh tất định với seed=42 từ tập mẫu âm của split đó,
     hoặc theo `group_ids` truyền vào).
   - Xử lý đồng điểm (tie-breaking): Quy tắc `conservative` (pessimistic) — nếu điểm dự đoán của
     mẫu âm >= mẫu dương thì mẫu âm đó đều xếp trên mẫu dương (tránh làm phồng Hits@K khi mô hình
     xuất điểm bằng nhau). Hỗ trợ thêm `average` và `optimistic`.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def _to_numpy(arr: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy().astype(np.float64).ravel()
    return np.asarray(arr, dtype=np.float64).ravel()


def compute_continuous_metrics(y_true, y_score) -> Dict[str, float]:
    """
    Tính các chỉ số từ điểm dự đoán liên tục (không phụ thuộc ngưỡng):
    - ROC-AUC
    - Average Precision (AP)
    """
    yt = _to_numpy(y_true)
    ys = _to_numpy(y_score)
    if len(np.unique(yt)) < 2:
        return {"roc_auc": 0.0, "ap": 0.0}
    return {
        "roc_auc": float(roc_auc_score(yt, ys)),
        "ap": float(average_precision_score(yt, ys)),
    }


def select_best_f1_threshold(
    val_y_true, val_y_score, num_grid: int = 199
) -> Tuple[float, float]:
    """
    Tìm ngưỡng `threshold` trong khoảng (0.01 .. 0.99) tối ưu hóa F1-Score trên tập Validation.
    Trả về `(best_threshold, best_val_f1)`.
    """
    yt = _to_numpy(val_y_true)
    ys = _to_numpy(val_y_score)
    thresholds = np.linspace(0.05, 0.95, num_grid)
    best_th = 0.5
    best_f1 = -1.0
    for th in thresholds:
        preds = (ys >= th).astype(np.int32)
        score = f1_score(yt, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_th = float(th)
    return best_th, best_f1


def compute_f1_at_threshold(y_true, y_score, threshold: float) -> Dict[str, float]:
    """
    Tính F1-Score (kèm Precision, Recall) tại ngưỡng `threshold` cố định từ tập Validation.
    """
    yt = _to_numpy(y_true).astype(np.int32)
    ys = _to_numpy(y_score)
    preds = (ys >= threshold).astype(np.int32)
    return {
        "f1": float(f1_score(yt, preds, zero_division=0)),
        "precision": float(precision_score(yt, preds, zero_division=0)),
        "recall": float(recall_score(yt, preds, zero_division=0)),
        "threshold": float(threshold),
    }


def compute_hits_at_k(
    y_true,
    y_score,
    k: int = 10,
    group_ids: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_neg_candidates: int = 50,
    seed: int = 42,
    tie_breaker: str = "conservative",
) -> float:
    """
    Tính Hits@K theo quy tắc nhóm ứng viên chuẩn của nhóm:
    - Nếu truyền `group_ids`: Xếp hạng từng mẫu dương trong nhóm `group_id` tương ứng.
    - Nếu không truyền `group_ids` (mặc định chuẩn cho `val_pairs.pt` / `test_pairs.pt`):
      Mỗi cạnh dương thứ `i` được ghép với đúng `num_neg_candidates` (50) mẫu âm cố định
      được rút tất định (`np.random.RandomState(seed)`) từ toàn bộ mẫu âm của tập đánh giá.
    - Xử lý đồng điểm (`tie_breaker`):
      * `'conservative'` (mặc định): rank = 1 + số mẫu âm có điểm >= điểm mẫu dương.
      * `'optimistic'`: rank = 1 + số mẫu âm có điểm > điểm mẫu dương.
      * `'average'`: rank = 1 + số mẫu âm > điểm dương + 0.5 * số mẫu âm == điểm dương.
    """
    yt = _to_numpy(y_true)
    ys = _to_numpy(y_score)

    pos_idx = np.where(yt == 1)[0]
    neg_idx = np.where(yt == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return 0.0

    pos_scores = ys[pos_idx]
    neg_scores = ys[neg_idx]

    def _calc_rank(p_score: float, cand_neg_scores: np.ndarray) -> float:
        greater = np.sum(cand_neg_scores > p_score)
        equal = np.sum(cand_neg_scores == p_score)
        if tie_breaker == "optimistic":
            return 1.0 + float(greater)
        elif tie_breaker == "average":
            return 1.0 + float(greater) + 0.5 * float(equal)
        else:  # conservative (mặc định)
            return 1.0 + float(greater + equal)

    hits = []
    if group_ids is not None:
        gids = _to_numpy(group_ids).astype(np.int64)
        for idx in pos_idx:
            g = gids[idx]
            cand_mask = (yt == 0) & (gids == g)
            cand_neg = ys[cand_mask]
            if len(cand_neg) == 0:
                cand_neg = neg_scores
            rank = _calc_rank(ys[idx], cand_neg)
            hits.append(1.0 if rank <= k else 0.0)
    else:
        rng = np.random.RandomState(seed)
        n_neg = len(neg_scores)
        pool_size = min(num_neg_candidates, n_neg)
        for p_score in pos_scores:
            sampled_idx = rng.choice(n_neg, size=pool_size, replace=False)
            cand_neg = neg_scores[sampled_idx]
            rank = _calc_rank(p_score, cand_neg)
            hits.append(1.0 if rank <= k else 0.0)

    return float(np.mean(hits))


def evaluate_all(
    val_y_true,
    val_y_score,
    test_y_true,
    test_y_score,
    val_groups=None,
    test_groups=None,
    k: int = 10,
    num_neg_candidates: int = 50,
    scenario_name: str = "Overall",
    model_name: str = "GCN",
    seed: int = 42,
) -> Dict[str, Union[str, int, float]]:
    """
    Pipeline đánh giá chuẩn hóa từ đầu đến cuối (Bước 3 & Bước 5):
    1. Chọn ngưỡng F1 trên tập Validation.
    2. Giữ nguyên ngưỡng F1 đó để đánh giá trên tập Test.
    3. Tính ROC-AUC, AP, F1, Hits@K cho cả Validation và Test.
    4. Trả về dictionary chuẩn với các cột bảng báo cáo:
       `Scenario | Model | Seed | ROC-AUC | AP | F1 | Hits@10`
    """
    val_cont = compute_continuous_metrics(val_y_true, val_y_score)
    best_th, val_f1 = select_best_f1_threshold(val_y_true, val_y_score)
    val_f1_dict = compute_f1_at_threshold(val_y_true, val_y_score, best_th)
    val_hits = compute_hits_at_k(
        val_y_true,
        val_y_score,
        k=k,
        group_ids=val_groups,
        num_neg_candidates=num_neg_candidates,
    )

    test_cont = compute_continuous_metrics(test_y_true, test_y_score)
    test_f1_dict = compute_f1_at_threshold(test_y_true, test_y_score, best_th)
    test_hits = compute_hits_at_k(
        test_y_true,
        test_y_score,
        k=k,
        group_ids=test_groups,
        num_neg_candidates=num_neg_candidates,
    )

    return {
        "Scenario": scenario_name,
        "Model": model_name,
        "Seed": int(seed),
        "ROC-AUC": round(test_cont["roc_auc"], 4),
        "AP": round(test_cont["ap"], 4),
        "F1": round(test_f1_dict["f1"], 4),
        f"Hits@{k}": round(test_hits, 4),
        "Val_ROC-AUC": round(val_cont["roc_auc"], 4),
        "Val_AP": round(val_cont["ap"], 4),
        "Val_F1": round(val_f1_dict["f1"], 4),
        f"Val_Hits@{k}": round(val_hits, 4),
        "Best_Threshold": round(best_th, 4),
        "Test_Precision": round(test_f1_dict["precision"], 4),
        "Test_Recall": round(test_f1_dict["recall"], 4),
    }


def evaluate_by_negative_types(
    val_y_true,
    val_y_score,
    test_y_true,
    test_y_score,
    test_edge_type,
    model_name: str = "GCN",
    seed: int = 42,
    k: int = 10,
) -> pd.DataFrame:
    """
    Đánh giá chi tiết mô hình trên kịch bản tổng thể (`Overall`) và từng kịch bản
    phân lớp mẫu âm (`Type 1..5`) của Ku em 1, giữ cố định ngưỡng F1 đã chọn từ tập Validation.
    """
    best_th, _ = select_best_f1_threshold(val_y_true, val_y_score)
    yt = _to_numpy(test_y_true)
    ys = _to_numpy(test_y_score)
    et = _to_numpy(test_edge_type).astype(np.int64)

    scenarios = [
        ("Overall (All 5 Neg Types)", (yt == 1) | (yt == 0)),
        ("Scenario 1: AML-AML (Cross-Case)", (yt == 1) | (et == 1)),
        ("Scenario 2: AML-Clean", (yt == 1) | (et == 2)),
        ("Scenario 3: Suspected KYC", (yt == 1) | (et == 3)),
        ("Scenario 4: Shared IP/Device", (yt == 1) | (et == 4)),
        ("Scenario 5: Clean-Clean Random", (yt == 1) | (et == 5)),
    ]

    rows = []
    for sc_name, mask in scenarios:
        sub_y = yt[mask]
        sub_s = ys[mask]
        cont = compute_continuous_metrics(sub_y, sub_s)
        f1_d = compute_f1_at_threshold(sub_y, sub_s, best_th)
        hits = compute_hits_at_k(sub_y, sub_s, k=k, num_neg_candidates=50)
        rows.append(
            {
                "Scenario": sc_name,
                "Model": model_name,
                "Seed": int(seed),
                "ROC-AUC": round(cont["roc_auc"], 4),
                "AP": round(cont["ap"], 4),
                "F1": round(f1_d["f1"], 4),
                f"Hits@{k}": round(hits, 4),
                "Threshold": round(best_th, 4),
                "Num_Pos": int((sub_y == 1).sum()),
                "Num_Neg": int((sub_y == 0).sum()),
            }
        )
    return pd.DataFrame(rows)
