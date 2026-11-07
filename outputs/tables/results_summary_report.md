# Báo cáo Tổng hợp Kết quả Thực nghiệm — Phần việc Ku em 2 (GCN / GAT, Metrics & Case Study)

---

## 1. Thống kê Dữ liệu Đầu vào & Kiểm tra Chống Rò rỉ (Bước 1)

- **Nguồn dữ liệu bàn giao từ Ku em 1:** `data/processed/results/processed_data/` (`processed_graph.pt`, `train_pairs.pt`, `val_pairs.pt`, `test_pairs.pt`).
- **Đồ thị đồng nhất khách hàng (`Homogeneous Customer Graph`) cho GCN & GAT:**
  - Số nút (`num_nodes`): **50,000** khách hàng (`CUST_00000` .. `CUST_49999`).
  - Kích thước đặc trưng nút (`x.shape`): **`[50000, 60]`** (`float32`, `0 NaN`, `0 Inf`), kết hợp đặc trưng gốc `customer` (10 chiều), đặc trưng tổng hợp `account` (13 chiều), `device` (9 chiều), `ip` (7 chiều), `transacts` (5 chiều) và vector nhúng lân cận cấu trúc (16 chiều).
  - Tổng số cạnh truyền thông điệp (`edge_index`, 2 chiều): **370,414** cạnh có hướng sau hợp nhất và khử trùng lặp:
    - `RELATED_TO (KYC)`: `62,262` cạnh
    - `SHARED_DEVICE`: `2,676` cạnh
    - `SHARED_IP`: `3,218` cạnh
    - `TRANSACTS (Giao dịch lớn/nhiều lần)`: `306,892` cạnh
- **Kiểm tra chống rò rỉ dữ liệu (`check_edge_leakage`):**
  - Tỷ lệ cạnh đồng phạm dương (`y=1`) của `Train / Val / Test` xuất hiện trong quan hệ khai báo `RELATED_TO`: **0 / 9,193 (`0.00%`)**.
  - Giao nhau tập đỉnh đồng phạm dương giữa `Train ∩ Val`, `Train ∩ Test`, `Val ∩ Test`: **0 đỉnh (`0.00%`)** — đảm bảo đánh giá quy nạp theo kịch bản vụ án (`Scenario Split`).

| Tập dữ liệu | Tổng số cặp | Mẫu Dương (`y=1`) | Mẫu Âm (`y=0`) | Type 1: AML-AML | Type 2: AML-Clean | Type 3: Suspected | Type 4: Shared IP/Dev | Type 5: Clean-Clean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN (70%)** | 12,664 | 6,332 | 6,332 | 1,899 | 949 | 949 | 496 | 2,039 |
| **VAL (15%)** | 2,302 | 1,151 | 1,151 | 345 | 172 | 172 | 17 | 445 |
| **TEST (15%)** | 3,420 | 1,710 | 1,710 | 513 | 256 | 256 | 32 | 653 |

---

## 2. Bảng Kết quả Thực nghiệm Chính thức (Bước 2 → Bước 5)

### 2.1. Tổng hợp `Mean ± Std` trên 3 Seeds (`42, 43, 44`)

| Scenario | Model | ROC-AUC | AP | F1 | Hits@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall (All 5 Neg Types)** | **GCN** | **0.8113 ± 0.0050** | **0.7976 ± 0.0080** | **0.7558 ± 0.0038** | **0.6704 ± 0.0095** |
| **Overall (All 5 Neg Types)** | **GAT** | 0.8000 ± 0.0005 | 0.7918 ± 0.0049 | 0.7383 ± 0.0043 | 0.6477 ± 0.0039 |
| **Scenario 1: AML-AML (Cross-Case)** | **GCN** | **0.6020 ± 0.0075** | **0.8386 ± 0.0049** | **0.7896 ± 0.0044** | 0.3396 ± 0.0120 |
| **Scenario 1: AML-AML (Cross-Case)** | **GAT** | 0.5934 ± 0.0079 | 0.8367 ± 0.0070 | 0.7690 ± 0.0058 | **0.3462 ± 0.0176** |
| **Scenario 2: AML-Clean** | **GCN** | **0.8493 ± 0.0038** | **0.9761 ± 0.0007** | **0.8628 ± 0.0008** | **0.7526 ± 0.0134** |
| **Scenario 2: AML-Clean** | **GAT** | 0.8314 ± 0.0149 | 0.9731 ± 0.0027 | 0.8382 ± 0.0076 | 0.7189 ± 0.0105 |
| **Scenario 3: Suspected KYC** | **GCN** | **0.9206 ± 0.0002** | **0.9875 ± 0.0004** | **0.8790 ± 0.0027** | **0.8575 ± 0.0064** |
| **Scenario 3: Suspected KYC** | **GAT** | 0.9152 ± 0.0035 | 0.9869 ± 0.0010 | 0.8544 ± 0.0050 | 0.8536 ± 0.0131 |
| **Scenario 4: Shared IP/Device** | **GCN** | **0.8306 ± 0.0064** | 0.9930 ± 0.0013 | **0.8823 ± 0.0043** | **0.9045 ± 0.0129** |
| **Scenario 4: Shared IP/Device** | **GAT** | 0.8125 ± 0.0092 | **0.9931 ± 0.0003** | 0.8561 ± 0.0061 | 0.8215 ± 0.0580 |
| **Scenario 5: Clean-Clean Random** | **GCN** | **0.9170 ± 0.0095** | **0.9709 ± 0.0035** | **0.8708 ± 0.0036** | **0.8571 ± 0.0206** |
| **Scenario 5: Clean-Clean Random** | **GAT** | 0.9041 ± 0.0012 | 0.9660 ± 0.0007 | 0.8461 ± 0.0043 | 0.8298 ± 0.0027 |

### 2.2. Kết quả từng Seed trên Kịch bản Tổng thể (`Overall`)

| Scenario | Model | Seed | ROC-AUC | AP | F1 | Hits@10 | Val F1 Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Overall (All 5 Neg Types) | GCN | 42 | 0.8152 | 0.8060 | 0.7586 | 0.6813 | 0.3227 |
| Overall (All 5 Neg Types) | GCN | 43 | 0.8056 | 0.7901 | 0.7515 | 0.6649 | 0.3955 |
| Overall (All 5 Neg Types) | GCN | 44 | 0.8131 | 0.7968 | 0.7573 | 0.6649 | 0.3136 |
| Overall (All 5 Neg Types) | GAT | 42 | 0.7999 | 0.7904 | 0.7343 | 0.6520 | 0.4455 |
| Overall (All 5 Neg Types) | GAT | 43 | 0.8005 | 0.7878 | 0.7429 | 0.6444 | 0.3818 |
| Overall (All 5 Neg Types) | GAT | 44 | 0.7996 | 0.7973 | 0.7378 | 0.6468 | 0.4682 |

---

## 3. Phân tích Case Study trên Tập Kiểm tra (Bước 6)

| Trường hợp | Cặp nút kiểm tra | Nhãn thật (`y`) | Phân loại mẫu | GCN Score | GCN Rank | GAT Score | GAT Rank | Ngưỡng F1 (Val) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Case 1: Dự đoán ĐÚNG (True Positive)** | `(CUST_00644, CUST_08708)` | `1` | Đồng phạm cùng kịch bản | `0.8963` | `1/51` | `0.9468` | `1/51` | `0.4455` |
| **Case 2: Dự đoán SAI (False Positive)** | `(CUST_04325, CUST_35098)` | `0` | Type 1: AML-AML (Khác kịch bản) | `0.8426` | `1/51` | `0.9258` | `1/51` | `0.4455` |

### Nhận xét chuyên môn:
1. **Điểm mạnh của GCN & GAT (Case 1 — `(CUST_00644, CUST_08708)`):**
   - Hai đối tượng hoàn toàn ẩn danh trên hồ sơ KYC (`RELATED_TO = 0`), nhưng cùng chia sẻ một cụm lân cận dày đặc qua kênh thiết bị (`SHARED_DEVICE`: `CUST_23409`, `CUST_26694`, `CUST_46810`, `CUST_47477`, `CUST_06806`).
   - Nhờ lan truyền thông điệp qua các nút trung gian này, cả GCN (`0.8963`) và GAT (`0.9468`) đều xếp cặp đồng phạm này ở vị trí **Top 1/51 (`Rank 1/51`)**. Trên đồ thị Attention của GAT, trọng số chú ý $\alpha \in [0.11, 0.15]$ tập trung mạnh vào các nút trung gian dùng chung thiết bị.
2. **Giới hạn của mô hình đồng nhất thuần (Case 2 — `(CUST_04325, CUST_35098)` & Scenario 1):**
   - Trên kịch bản **Type 1 (`AML-AML` khác vụ án)**, cả hai đầu mút đều là tội phạm rửa tiền thật nên mang đặc trưng hành vi bất thường rất giống nhau, dù vùng mạng lân cận của họ hoàn toàn rời nhau (không có lân cận chung).
   - Khi chiếu đồ thị dị thể về đồ thị đồng nhất, GCN và GAT đạt `ROC-AUC ~ 0.60` trên riêng nhóm `Type 1` (so với `0.83 – 0.92` trên `Type 2..5`), dẫn tới hiện tượng dương tính giả giữa hai tội phạm khác băng nhóm. Đây là cơ sở thực nghiệm rõ ràng để nhóm phát triển thêm mô hình đồ thị dị thể (HeteroGNN) hoặc bộ giải mã khoảng cách cấu trúc ở phần mở rộng.

