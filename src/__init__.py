"""
Gói tiện ích dùng chung cho phần việc của Ku em 2 (Baseline GCN/GAT, Metrics & Visualization):
- `data_validator`: Nạp dữ liệu từ Ku em 1, tổng hợp đồ thị đồng nhất khách hàng & kiểm tra chống rò rỉ.
- `models`: Định nghĩa `GCNEncoder`, `GATEncoder`, `LinkDecoder`, `GNNLinkPredictor` và hàm huấn luyện.
- `metrics`: Bộ tính `ROC-AUC`, `AP`, `F1` (chọn ngưỡng trên Val) và `Hits@K` dùng chung cho toàn nhóm.
- `viz_utils`: Vẽ đường cong Loss/Validation AP, biểu đồ so sánh đa kịch bản và Case Study (kèm GAT Attention).
"""

from .data_validator import (
    DEFAULT_PROCESSED_DIR,
    EDGE_CHANNEL_NAMES,
    NEGATIVE_TYPE_NAMES,
    build_homogeneous_customer_graph,
    check_edge_leakage,
    inspect_graph_data,
    load_split_pairs,
)
from .metrics import (
    compute_continuous_metrics,
    compute_f1_at_threshold,
    compute_hits_at_k,
    evaluate_all,
    evaluate_by_negative_types,
    select_best_f1_threshold,
)
from .models import (
    GATConv,
    GATEncoder,
    GCNConv,
    GCNEncoder,
    GNNLinkPredictor,
    LinkDecoder,
    set_seed,
    train_and_evaluate_model,
)
from .viz_utils import (
    extract_local_subgraph,
    plot_case_study_figure,
    plot_gat_attention_subgraph,
    plot_scenario_comparison,
    plot_training_curves,
    select_case_study_pairs,
)
