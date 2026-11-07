"""
Module định nghĩa kiến trúc Mô hình (Bước 2 & Bước 4):
1. `GCNEncoder`: Bộ mã hóa GCN thuần (2 lớp `GCNConv` theo Kipf & Welling, ICLR 2017).
2. `GATEncoder`: Bộ mã hóa GAT thuần (2 lớp `GATConv` theo Veličković et al., ICLR 2018 với Multi-Head Attention
   và hỗ trợ trích xuất trọng số Attention `return_attention_weights=True`).
3. `LinkDecoder`: Bộ giải mã cặp nút nhận biểu diễn `(z_u, z_v)` và trả về 1 logit liên kết.
4. `GNNLinkPredictor`: Mô hình ghép nối `Encoder + LinkDecoder`, dùng chung quy trình huấn luyện (`BCEWithLogitsLoss`),
   lưu/nạp checkpoint theo `Validation AP` cho cả `GCN` và `GAT`.
"""

import copy
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import compute_continuous_metrics


def set_seed(seed: int = 42) -> None:
    """Cố định toàn bộ random seed để đảm bảo tính tái lập giữa các lần chạy."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GCNConv(nn.Module):
    """
    Lớp Graph Convolutional Network chuẩn (Kipf & Welling, 2017):
    H' = D~^{-1/2} A~ D~^{-1/2} X W + b, với A~ = A + I (thêm self-loop).
    Tương thích hoàn toàn với API của PyTorch Geometric và chạy trực tiếp trên PyTorch thuần.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def compute_norm(edge_index: torch.Tensor, num_nodes: int, device: torch.device, dtype: torch.dtype):
        loop_nodes = torch.arange(num_nodes, device=device, dtype=edge_index.dtype)
        loop_edge = torch.stack([loop_nodes, loop_nodes], dim=0)
        ei = torch.cat([edge_index, loop_edge], dim=1)

        row, col = ei[0], ei[1]
        deg = torch.zeros(num_nodes, device=device, dtype=dtype)
        deg.index_add_(0, col, torch.ones(col.shape[0], device=device, dtype=dtype))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0.0)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return ei, norm

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        cached_norm: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        h = self.lin(x)

        if cached_norm is not None:
            ei, norm = cached_norm
        else:
            ei, norm = self.compute_norm(edge_index, num_nodes, x.device, x.dtype)

        row, col = ei[0], ei[1]
        msg = h[row] * norm.unsqueeze(-1)
        out = torch.zeros_like(h)
        out.index_add_(0, col, msg)

        if self.bias is not None:
            out = out + self.bias
        return out


class GATConv(nn.Module):
    """
    Lớp Graph Attention Network chuẩn (Veličković et al., ICLR 2018):
    e_{ij} = LeakyReLU(a_src^T W h_i + a_dst^T W h_j)
    alpha_{ij} = softmax_j(e_{ij})
    Hỗ trợ Multi-Head Attention (`heads`, `concat=True/False`) và `return_attention_weights=True`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        out_dim = heads * out_channels if concat else out_channels
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ):
        num_nodes = x.size(0)
        H, C = self.heads, self.out_channels

        # Thêm self-loops để nút luôn chú ý tới chính nó
        loop_nodes = torch.arange(num_nodes, device=x.device, dtype=edge_index.dtype)
        loop_edge = torch.stack([loop_nodes, loop_nodes], dim=0)
        ei = torch.cat([edge_index, loop_edge], dim=1)
        row, col = ei[0], ei[1]  # truyền tin từ row -> col

        x_proj = self.lin(x).view(num_nodes, H, C)
        alpha_src = (x_proj * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_proj * self.att_dst).sum(dim=-1)  # [N, H]

        e = F.leaky_relu(alpha_src[row] + alpha_dst[col], negative_slope=self.negative_slope)  # [E, H]

        # Segment softmax theo nút đích `col` (ổn định số học với amax)
        e_max = torch.full((num_nodes, H), -1e9, device=x.device, dtype=x.dtype)
        e_max.scatter_reduce_(0, col.unsqueeze(-1).expand(-1, H), e, reduce="amax", include_self=True)
        exp_e = torch.exp(e - e_max[col])

        denom = torch.zeros((num_nodes, H), device=x.device, dtype=x.dtype)
        denom.index_add_(0, col, exp_e)
        alpha = exp_e / (denom[col] + 1e-12)  # [E, H]
        alpha_drop = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = x_proj[row] * alpha_drop.unsqueeze(-1)  # [E, H, C]
        out = torch.zeros((num_nodes, H, C), device=x.device, dtype=x.dtype)
        out.index_add_(0, col, msg)

        if self.concat:
            out = out.view(num_nodes, H * C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, (ei, alpha)
        return out


class GCNEncoder(nn.Module):
    """
    Bộ mã hóa GCN thuần (Bước 2):
    Đồ thị đầu vào `(x, edge_index)` -> 2 lớp `GCNConv` -> Vector biểu diễn từng nút `z` [N, out_channels].
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout = dropout

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Linear(in_channels, out_channels)
        self._cached_edge_ptr = None
        self._cached_norm = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self._cached_edge_ptr != edge_index.data_ptr():
            self._cached_norm = GCNConv.compute_norm(edge_index, x.size(0), x.device, x.dtype)
            self._cached_edge_ptr = edge_index.data_ptr()

        h = self.conv1(x, edge_index, cached_norm=self._cached_norm)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, cached_norm=self._cached_norm)
        h = self.bn2(h)
        z = h + self.skip(x)
        return z


class GATEncoder(nn.Module):
    """
    Bộ mã hóa GAT thuần (Bước 4):
    Thay `GCNConv` bằng `GATConv` với cơ chế Multi-Head Attention.
    - Lớp 1: `heads` đầu chú ý, ghép nối (`concat=True`) -> kích thước `hidden_channels * heads`.
    - Lớp 2: `heads` đầu chú ý, lấy trung bình (`concat=False`) -> kích thước `out_channels`
      khớp hoàn toàn với đầu vào của `LinkDecoder`.
    - Hỗ trợ `encode_with_attention` để trích xuất trọng số Attention phục vụ Case Study (Bước 6).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 16,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)

        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.skip = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        z = h + self.skip(x)
        return z

    @torch.no_grad()
    def encode_with_attention(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Trả về `(z, (edge_idx_1, alpha_1), (edge_idx_2, alpha_2))` để minh họa Attention ở Bước 6.
        """
        self.eval()
        h, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        h = self.bn1(h)
        h = F.elu(h)
        h2, attn2 = self.conv2(h, edge_index, return_attention_weights=True)
        h2 = self.bn2(h2)
        z = h2 + self.skip(x)
        return z, attn1, attn2


class LinkDecoder(nn.Module):
    """
    Bộ giải mã cho cặp nút `(u, v)` (Bước 2 & Bước 4):
    Nhận vector biểu diễn của 2 nút `(z_u, z_v)` và trả về 1 logit liên kết (chưa qua sigmoid).
    """

    def __init__(self, in_channels: int = 64, hidden_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        pair_dim = in_channels * 3 + 1
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        z_u = z[edge_label_index[0]]
        z_v = z[edge_label_index[1]]

        hadamard = z_u * z_v
        diff = torch.abs(z_u - z_v)
        summ = z_u + z_v
        cos = F.cosine_similarity(z_u, z_v, dim=-1, eps=1e-8).unsqueeze(-1)

        pair_feat = torch.cat([hadamard, diff, summ, cos], dim=-1)
        logits = self.mlp(pair_feat).squeeze(-1)
        return logits


class GNNLinkPredictor(nn.Module):
    """
    Mô hình tổng thể kết hợp Encoder (`GCN` hoặc `GAT`) và `LinkDecoder`.
    """

    def __init__(
        self,
        model_type: str,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.model_type = model_type.upper()
        self.config = {
            "model_type": self.model_type,
            "in_channels": int(in_channels),
            "hidden_channels": int(hidden_channels),
            "out_channels": int(out_channels),
            "heads": int(heads) if self.model_type == "GAT" else 1,
            "dropout": float(dropout),
        }

        if self.model_type == "GCN":
            self.encoder = GCNEncoder(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                dropout=dropout,
            )
        elif self.model_type == "GAT":
            gat_hidden = max(8, hidden_channels // heads)
            self.encoder = GATEncoder(
                in_channels=in_channels,
                hidden_channels=gat_hidden,
                out_channels=out_channels,
                heads=heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"model_type không hợp lệ: {model_type}. Chọn 'GCN' hoặc 'GAT'.")

        self.decoder = LinkDecoder(
            in_channels=out_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_label_index)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)

    @torch.no_grad()
    def predict_proba(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Chỉ dùng `torch.sigmoid` khi cần chuyển logit thành điểm dự đoán trong khoảng [0, 1] để báo cáo.
        """
        self.eval()
        logits = self.forward(x, edge_index, edge_label_index)
        return torch.sigmoid(logits)


def train_and_evaluate_model(
    model_type: str,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    splits: Dict[str, Dict[str, torch.Tensor]],
    seed: int = 42,
    epochs: int = 80,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
    hidden_channels: int = 64,
    out_channels: int = 64,
    heads: int = 4,
    dropout: float = 0.2,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    """
    Quy trình huấn luyện và chọn checkpoint thống nhất cho cả GCN và GAT (Bước 2, 4, 5):
    1. Dùng `BCEWithLogitsLoss` cho nhãn nhị phân.
    2. Đánh giá Validation mỗi epoch và lưu checkpoint tốt nhất theo **Validation AP**.
    3. Sau khi huấn luyện xong, nạp lại checkpoint tốt nhất để xuất điểm dự đoán cho Validation & Test.
    """
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    model = GNNLinkPredictor(
        model_type=model_type,
        in_channels=x.shape[1],
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        dropout=dropout,
    ).to(dev)

    x_dev = x.to(dev)
    ei_dev = edge_index.to(dev)

    train_eli = splits["train"]["edge_label_index"].to(dev)
    train_y = splits["train"]["edge_label"].float().to(dev)
    val_eli = splits["val"]["edge_label_index"].to(dev)
    val_y = splits["val"]["edge_label"].float().to(dev)
    test_eli = splits["test"]["edge_label_index"].to(dev)
    test_y = splits["test"]["edge_label"].float().to(dev)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_ap = -1.0
    best_epoch = 0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_dev, ei_dev, train_eli)
        loss = criterion(logits, train_y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss không hữu hạn tại epoch {epoch}: {loss.item()}")
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(x_dev, ei_dev)
            val_logits = model.decode(z, val_eli)
            val_loss = criterion(val_logits, val_y).item()
            val_scores = torch.sigmoid(val_logits).cpu().numpy()
            val_metrics = compute_continuous_metrics(val_y.cpu().numpy(), val_scores)
            val_ap = val_metrics["ap"]
            val_auc = val_metrics["roc_auc"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "val_loss": float(val_loss),
                "val_ap": float(val_ap),
                "val_roc_auc": float(val_auc),
            }
        )

        if val_ap > best_val_ap:
            best_val_ap = float(val_ap)
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            print(
                f"[{model_type.upper()} | Seed {seed}] Epoch {epoch:03d}/{epochs:03d} | "
                f"Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AP: {val_ap:.4f} (Best: {best_val_ap:.4f} @ Ep {best_epoch})"
            )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        z_best = model.encode(x_dev, ei_dev)
        val_scores_best = torch.sigmoid(model.decode(z_best, val_eli)).cpu().numpy()
        test_scores_best = torch.sigmoid(model.decode(z_best, test_eli)).cpu().numpy()

    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "config": model.config,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_ap": best_val_ap,
            },
            checkpoint_path,
        )

    return {
        "model": model,
        "config": model.config,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_ap": best_val_ap,
        "history": history,
        "val_y_true": val_y.cpu().numpy(),
        "val_y_score": val_scores_best,
        "test_y_true": test_y.cpu().numpy(),
        "test_y_score": test_scores_best,
    }
