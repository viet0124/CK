# Biên bản Thống nhất Dữ liệu & Quy ước Đánh giá (Ku em 1 ↔ Ku em 2)

Tài liệu này ghi nhận kết quả kiểm tra thực tế trên bộ dữ liệu đã tiền xử lý từ **Ku em 1** tại `data/processed/results/processed_data/`.

---

## 1. Bảng thành phần dữ liệu đã nhận và kiểm tra (Bước 1 — PASSED)

| Tên trường / File | Kiểu dữ liệu & Kích thước thực tế | Mục đích sử dụng | Trạng thái |
| :--- | :--- | :--- | :---: |
| `processed_graph.pt` (`node_features`) | 6 loại đỉnh: `customer (50,000 × 10)`, `account (112,620 × 6)`, `device (167,286 × 6)`, `ip (166,933 × 4)`, `beneficiary (644,166 × 8)`, `bank (4 × 4)` | Tổng hợp thành ma trận đặc trưng đồng nhất `x` (`FloatTensor [50000, 60]`), `0 NaN`, `0 Inf` | [x] |
| `processed_graph.pt` (`edge_index`) | 7 quan hệ 2 chiều (`OWNS`, `BANKS_WITH`, `USES_DEVICE`, `USES_IP`, `TRANSACTS`, `WIRES_TO`, `RELATED_TO`) | Chiếu về đồ thị đồng nhất khách hàng `edge_index` (`LongTensor [2, 370414]`) cho GCN/GAT | [x] |
| `node_id_map` | `dict` (`0..49999 <-> "CUST_00000".."CUST_49999"`) | Hiển thị đúng định danh đối tượng khi vẽ đồ thị Case Study | [x] |
| `train_pairs.pt` | `edge_label_index [2, 12664]`, `edge_label [12664]`, `edge_type [12664]` | `6,332` mẫu dương (`y=1`) + `6,332` mẫu âm (`y=0`, gồm 5 phân lớp `Type 1..5`) | [x] |
| `val_pairs.pt` | `edge_label_index [2, 2302]`, `edge_label [2302]`, `edge_type [2302]` | `1,151` mẫu dương (`y=1`) + `1,151` mẫu âm (`y=0`) — dùng chọn checkpoint theo `Val AP` và ngưỡng `F1` | [x] |
| `test_pairs.pt` | `edge_label_index [2, 3420]`, `edge_label [3420]`, `edge_type [3420]` | `1,710` mẫu dương (`y=1`) + `1,710` mẫu âm (`y=0`) — dùng đánh giá cuối cùng | [x] |

---

## 2. Xác minh Chống Rò rỉ Dữ liệu (Data Leakage Verification)

1. **Bản chất cạnh trong `edge_index` vs Quan hệ cần dự đoán (`y=1`)**:
   - Quan hệ mục tiêu cần dự đoán (`y=1` từ `labels.json`) là **quan hệ đồng phạm rửa tiền (`AML Accomplice`)** trong 297 kịch bản tội phạm tài chính AusAML.
   - Kết quả kiểm tra tự động (`check_edge_leakage`): **0 / 9,193 (`0.00%`)** cặp đồng phạm dương trong `Train`, `Val`, `Test` xuất hiện trong quan hệ khai báo `RELATED_TO` (cả chiều xuôi lẫn chiều ngược).
   - Các kênh truyền tin trong `edge_index` chỉ bao gồm quan hệ khai báo công khai (`RELATED_TO`: `62,262` cạnh) và các kênh hành vi quan sát được (`SHARED_DEVICE`: `2,676` cạnh, `SHARED_IP`: `3,218` cạnh, `TRANSACTS` giá trị cao/nhiều lần: `306,892` cạnh).
2. **Xác minh Scenario Split (Chia theo kịch bản vụ án)**:
   - Giao nhau về tập đỉnh đồng phạm dương (`y=1`) giữa `Train ∩ Val`, `Train ∩ Test`, và `Val ∩ Test` đều bằng **0** (`100%` phân tách theo kịch bản vụ án).
3. **Cơ cấu 5 phân lớp mẫu âm (`edge_type = 1..5`)**:
   - `Type 1 (30%)`: **AML–AML (Khác kịch bản)** — Hai đối tượng rửa tiền thật nhưng thuộc 2 vụ án khác nhau (chống mô hình học lối tắt chỉ dựa trên đặc trưng nút đơn lẻ).
   - `Type 2 (15%)`: **AML–Clean** — Một đối tượng tội phạm ghép với một khách hàng bình thường.
   - `Type 3 (15%)`: **Suspected KYC** — Khách hàng bị gắn cờ nghi vấn giả trong hồ sơ KYC.
   - `Type 4`: **Shared IP/Device** — Hai khách hàng sạch vô tình dùng chung mạng IP/thiết bị (Hard Negatives).
   - `Type 5`: **Clean–Clean Random** — Hai khách hàng bình thường ngẫu nhiên.

---

## 3. Quy ước tính `Hits@K` dùng chung cho toàn nhóm (`src/metrics.py`)

- **Nhóm ứng viên (`Candidate Grouping`)**:
  - Mỗi cạnh dương thực sự $e^+ = (u, v)$ trong tập đánh giá được xếp hạng cùng với một nhóm ứng viên gồm **$M = 50$ mẫu âm** cố định (được rút tất định với `RandomState(seed=42)` từ tập mẫu âm của split đó, hoặc truyền trực tiếp qua `group_ids`). Tổng kích thước nhóm xếp hạng là `51` ứng viên (`1 dương + 50 âm`, mức ngẫu nhiên kỳ vọng của `Hits@10` là $10/51 \approx 19.61\%$).
- **Quy tắc xử lý đồng điểm (`Tie-breaking = Conservative`)**:
  - Thứ hạng của cạnh dương được tính theo công thức bảo thủ:
    $$\text{Rank}(e^+) = 1 + \sum_{e^- \in \mathcal{C}_{neg}} \mathbb{I}\big(\text{score}(e^-) \ge \text{score}(e^+)\big)$$
  - Nếu `Rank(e+) <= K` (với `K = 10`), tính là `1` lần trúng (`Hit`), ngược lại là `0`.
