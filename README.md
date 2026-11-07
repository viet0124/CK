# Dự án Dự đoán Liên kết Đồng phạm (AusAML / Orange Bank) — Phần việc Ku em 2

Kho mã nguồn này hoàn thiện toàn bộ nhiệm vụ của **Ku em 2 (Baseline GCN/GAT, Metrics & Visualization)** theo đúng quy trình 7 bước, chạy trực tiếp trên dữ liệu tiền xử lý từ **Ku em 1** (`data/processed/results/processed_data/`).

---

## 1. Cấu trúc thư mục & Danh mục bàn giao (Bước 7)

```text
CK\
├── README.md                                   # Hướng dẫn chạy và tổng quan dự án
├── requirements.txt                            # Thư viện cần thiết
├── docs/
│   └── data_contract_ku_em_1.md                # Biên bản đối chiếu dữ liệu với Ku em 1 & Quy ước Hits@10
├── data/
│   └── processed/results/processed_data/       # Dữ liệu chuẩn do Ku em 1 bàn giao
│       ├── processed_graph.pt                  # Đồ thị 6 loại đỉnh, 7 quan hệ 2 chiều
│       ├── train_pairs.pt                      # 12,664 cặp (6,332 dương + 6,332 âm gồm Type 1..5)
│       ├── val_pairs.pt                        # 2,302 cặp (1,151 dương + 1,151 âm)
│       └── test_pairs.pt                       # 3,420 cặp (1,710 dương + 1,710 âm)
├── src/                                        # Các module dùng chung cho toàn nhóm
│   ├── __init__.py
│   ├── data_validator.py                       # Chuyển đổi đồ thị đồng nhất khách hàng & Kiểm tra chống rò rỉ
│   ├── models.py                               # GCNEncoder, GATEncoder (có trích xuất Attention), LinkDecoder
│   ├── metrics.py                              # Bộ tính ROC-AUC, AP, F1 (chọn ngưỡng trên Val), Hits@K dùng chung
│   └── viz_utils.py                            # Vẽ biểu đồ Loss/Val AP, so sánh kịch bản & Case Study
├── notebooks/                                  # 4 Jupyter Notebooks đã thực thi hoàn chỉnh (có sẵn output)
│   ├── 01_data_inspection_and_mock.ipynb       # Bước 1: Kiểm tra dữ liệu Ku em 1 & Chống rò rỉ
│   ├── 02_gcn_baseline.ipynb                   # Bước 2 & 3: Mô hình GCN thuần từ đầu đến cuối + Bộ đánh giá
│   ├── 03_gat_and_experiments.ipynb            # Bước 4 & 5: GAT + Thực nghiệm 3 Seeds (42, 43, 44)
│   └── 04_case_study_visualization.ipynb       # Bước 6 & 7: Case Study (Đúng/Sai + GAT Attention)
└── outputs/                                    # Kết quả thực nghiệm bàn giao cho nhóm
    ├── checkpoints/                            # Model weights (.pt), cấu hình và ngưỡng F1 (.json)
    │   ├── gcn_baseline_best.pt
    │   ├── gcn_seed42.pt, gcn_seed43.pt, gcn_seed44.pt
    │   ├── gat_seed42.pt, gat_seed43.pt, gat_seed44.pt
    │   └── experiment_configs_and_thresholds.json
    ├── predictions/                            # Điểm dự đoán Val/Test từng seed (.pt)
    │   ├── gcn_seed42_predictions.pt .. gcn_seed44_predictions.pt
    │   └── gat_seed42_predictions.pt .. gat_seed44_predictions.pt
    ├── tables/                                 # Bảng tổng hợp kết quả CSV & Markdown
    │   ├── results_per_seed.csv
    │   ├── results_mean_std.csv
    │   ├── results_mean_std_formatted.csv
    │   ├── case_study_pairs_summary.csv
    │   └── results_summary_report.md
    └── figures/                                # Hình biểu đồ và Case Study độ phân giải cao
        ├── training_curves_loss_val_ap.png
        ├── scenario_comparison_bar.png
        ├── case_study_correct_vs_incorrect.png
        └── case_study_gat_attention.png
```

---

## 2. Hướng dẫn sử dụng Bộ Đánh giá Dùng chung cho Thành viên khác (`src/metrics.py`)

Mọi mô hình trong nhóm chỉ cần xuất điểm dự đoán liên tục `val_y_score` và `test_y_score` (mảng 1D tương ứng với thứ tự cặp trong `val_pairs.pt` và `test_pairs.pt`) và gọi:

```python
from src.metrics import evaluate_all, evaluate_by_negative_types

# 1. Xuất 1 dòng kết quả tổng thể (tự động chọn ngưỡng F1 trên Val và cố định trên Test)
row = evaluate_all(
    val_y_true=val_labels,
    val_y_score=my_model_val_scores,
    test_y_true=test_labels,
    test_y_score=my_model_test_scores,
    scenario_name="Overall",
    model_name="MyHeteroGNN",
    seed=42,
)

# 2. Xuất bảng kết quả chi tiết trên từng kịch bản mẫu âm (Scenario 1..5)
df_scenarios = evaluate_by_negative_types(
    val_y_true=val_labels,
    val_y_score=my_model_val_scores,
    test_y_true=test_labels,
    test_y_score=my_model_test_scores,
    test_edge_type=test_edge_types,
    model_name="MyHeteroGNN",
    seed=42,
)
```

---

## 3. Tóm tắt Kết quả Chính (`Mean ± Std` trên 3 Seeds: `42, 43, 44`)

| Scenario | Model | ROC-AUC | AP | F1 | Hits@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall (All 5 Neg Types)** | **GCN** | `0.8113 ± 0.0050` | `0.7976 ± 0.0080` | `0.7558 ± 0.0038` | `0.6704 ± 0.0095` |
| **Overall (All 5 Neg Types)** | **GAT** | `0.8000 ± 0.0005` | `0.7918 ± 0.0049` | `0.7383 ± 0.0043` | `0.6477 ± 0.0039` |
| **Scenario 1: AML-AML (Cross-Case)** | **GCN / GAT** | `0.6020 / 0.5934` | `0.8386 / 0.8367` | `0.7896 / 0.7690` | `0.3396 / 0.3462` |
| **Scenario 2: AML-Clean** | **GCN / GAT** | `0.8493 / 0.8314` | `0.9761 / 0.9731` | `0.8628 / 0.8382` | `0.7526 / 0.7189` |
| **Scenario 3: Suspected KYC** | **GCN / GAT** | `0.9206 / 0.9152` | `0.9875 / 0.9869` | `0.8790 / 0.8544` | `0.8575 / 0.8536` |
| **Scenario 4: Shared IP/Device** | **GCN / GAT** | `0.8306 / 0.8125` | `0.9930 / 0.9931` | `0.8823 / 0.8561` | `0.9045 / 0.8215` |
| **Scenario 5: Clean-Clean Random** | **GCN / GAT** | `0.9170 / 0.9041` | `0.9709 / 0.9660` | `0.8708 / 0.8461` | `0.8571 / 0.8298` |
