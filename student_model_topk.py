import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# ==============================================================================
# 1. 核心模块: SE-Block (注意力机制)
# ==============================================================================
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    作用：让模型自动学习“哪些通道更重要”。
    参数量：极少 (约几千个参数)
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==============================================================================
# 2. 空间编码器 (Spatial Encoder)
# ==============================================================================
class SpatialEncoder(nn.Module):
    def __init__(self, in_channels: int = 5, out_channels: int = 96):
        super().__init__()

        # 目标：输出 32x32 的特征图 (如果输入是 128x128，则需要缩小 4 倍)
        self.features = nn.Sequential(
            # Layer 1: [128x128] -> [64x64] (Stride 2)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: [64x64] -> [32x32] (Stride 2)
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 3: [32x32] -> [32x32] (Stride 1)
            # 🔥 关键修改：stride=1 保持 32x32 分辨率，不再缩小！
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # 🔥 新增：SE-Block 注意力模块
            SEBlock(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ==============================================================================
# 3. MT-ConvLSTM (核心时序模块)
# ==============================================================================
class MTConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        self.W_t = nn.Parameter(torch.Tensor(hidden_channels))
        self.b_t = nn.Parameter(torch.Tensor(hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        nn.init.ones_(self.W_t)
        nn.init.zeros_(self.b_t)

    def forward(self, x, h_prev, c_prev, time_interval, motion_info):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)

        time_gate = (time_interval * motion_info).view(-1, 1, 1, 1)
        W_t = self.W_t.view(1, -1, 1, 1)
        b_t = self.b_t.view(1, -1, 1, 1)
        time_modulation = torch.sigmoid(time_gate * W_t + b_t)

        i = torch.sigmoid(i) * time_modulation
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class MTConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(MTConvLSTMCell(in_ch, hidden_channels, kernel_size))

    def forward(self, x, time_intervals, motion_info, lengths):
        B, K, C, H, W = x.shape
        device = x.device
        h = [torch.zeros(B, self.hidden_channels, H, W, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_channels, H, W, device=device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(K):
            x_t = x[:, t]
            time_t = time_intervals[:, t]
            motion_t = motion_info[:, t]
            mask = (t < lengths).float().view(B, 1, 1, 1)

            layer_input = x_t
            for layer_idx in range(self.num_layers):
                h_new, c_new = self.cells[layer_idx](layer_input, h[layer_idx], c[layer_idx], time_t, motion_t)
                h[layer_idx] = h_new * mask + h[layer_idx] * (1 - mask)
                c[layer_idx] = c_new * mask + c[layer_idx] * (1 - mask)
                layer_input = h[layer_idx]
            outputs.append(h[-1])

        return torch.stack(outputs, dim=1)


# ==============================================================================
# 3.5 Standard ConvLSTM (Ablation Baseline — no time modulation)
# ==============================================================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c_prev + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(in_ch, hidden_channels, kernel_size))

    def forward(self, x, time_intervals, motion_info, lengths):
        # time_intervals & motion_info accepted but ignored (standard LSTM)
        B, K, C, H, W = x.shape
        device = x.device
        h = [torch.zeros(B, self.hidden_channels, H, W, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_channels, H, W, device=device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(K):
            x_t = x[:, t]
            mask = (t < lengths).float().view(B, 1, 1, 1)
            layer_input = x_t
            for layer_idx in range(self.num_layers):
                h_new, c_new = self.cells[layer_idx](layer_input, h[layer_idx], c[layer_idx])
                h[layer_idx] = h_new * mask + h[layer_idx] * (1 - mask)
                c[layer_idx] = c_new * mask + c[layer_idx] * (1 - mask)
                layer_input = h[layer_idx]
            outputs.append(h[-1])

        return torch.stack(outputs, dim=1)


# ==============================================================================
# 4. 识别头 (Recognition Head)
# ==============================================================================
# 1. 修改 RecognitionHead，让它接收 dropout_rate
class RecognitionHead(nn.Module):
    # 🔥 新增 dropout_rate 参数
    def __init__(self, input_channels: int, spatial_size: int, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 🔥 这里不再写死 0.5，而是用传入的参数
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(input_channels, num_classes)

    def forward(self, x):
        B, K, C, H, W = x.shape
        x = x.view(B * K, C, H, W)
        x = self.gap(x)
        x = x.flatten(1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits.view(B, K, -1)


# ==============================================================================
# 5. 关键帧选择器 (KeyFrame Selector) - 最终论文版 Adaptive Top-K
# ==============================================================================
class KeyFrameSelector:
    """
    🔥 论文 SOTA 版：自适应 Top-K 选择器 (Adaptive Top-K Selector)

    核心逻辑：
    1. 计算动态阈值 (如：均值的 50%)。
    2. 统计高于阈值的帧数，并强制限制在 [min_frames, max_frames] 之间 (例如 8~15帧)。
    3. 根据算出的长度 L，提取 Top-L 个高能量帧，并恢复时间顺序。
    4. 针对 Batch>1 安全 Padding，用最后一帧补齐到 max_frames。
    """

    def __init__(self, min_frames=8, max_frames=15, diff_threshold=0.5):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.energy_ratio = diff_threshold

    def select_frames(self, voxel_grid: torch.Tensor):
        B, T, C, H, W = voxel_grid.shape
        device = voxel_grid.device

        # 1. 计算所有帧的能量
        frame_energy = voxel_grid.abs().sum(dim=[2, 3, 4])

        # 2. 自适应计算每个样本的"理想有效长度"
        mean_energy = frame_energy.mean(dim=1, keepdim=True)
        dynamic_threshold = mean_energy * self.energy_ratio
        valid_counts = (frame_energy > dynamic_threshold).sum(dim=1)

        # 🔥 核心：用 clamp 强行施加"下限(保底)"和"上限(防爆炸)"
        lengths = torch.clamp(valid_counts, min=self.min_frames, max=self.max_frames)

        # 3. 选出 Top-K (先粗略提取能量最高的 max_frames 帧)
        k_safe = min(self.max_frames, T)
        _, top_indices = torch.topk(frame_energy, k=k_safe, dim=1)

        target_len = self.max_frames
        final_indices = torch.zeros(B, target_len, dtype=torch.long, device=device)

        # 4. 精确截取与安全 Padding (完美兼容 Batch > 1)
        for b in range(B):
            L = lengths[b].item()

            # A. 截取前 L 个高能量帧
            indices = top_indices[b, :L]

            # B. 重新排序，恢复时间先后顺序 (LSTM的硬要求)
            sorted_indices, _ = torch.sort(indices)

            # C. 填入最终索引
            final_indices[b, :L] = sorted_indices

            # D. 安全补齐 (Padding)：不足 max_frames 的部分用最后一个有效帧重复
            if L < target_len:
                final_indices[b, L:] = sorted_indices[-1]

        # 5. Gather 提取真实张量并计算时间戳
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, target_len)
        padded_frames = voxel_grid[batch_idx, final_indices]
        time_intervals = final_indices.float() / T

        return padded_frames, lengths, time_intervals



# ==============================================================================
# 6. 学生模型主类 (适配 Dense-to-Sparse)
# ==============================================================================
class StudentModel(nn.Module):
    def __init__(self, encoder_channels=96, lstm_hidden=64, lstm_layers=1,
                 num_classes=11, min_keyframes=3, max_keyframes=15,
                 diff_threshold=0.05, teacher_dim=768,
                 dropout=0.5, use_md_lstm=True):   # ← add use_md_lstm
        super().__init__()

        self.selector = KeyFrameSelector(min_keyframes, max_keyframes, diff_threshold)
        self.encoder = SpatialEncoder(in_channels=5, out_channels=encoder_channels)

        # ← swap here
        if use_md_lstm:
            self.mt_lstm = MTConvLSTM(encoder_channels, lstm_hidden, lstm_layers)
        else:
            self.mt_lstm = ConvLSTM(encoder_channels, lstm_hidden, lstm_layers)

        self.head = RecognitionHead(lstm_hidden, spatial_size=32, num_classes=num_classes, dropout_rate=dropout)
        self.feature_adapter = nn.Linear(lstm_hidden, teacher_dim)

    def calculate_motion_info(self, voxel_frames):
        B = voxel_frames.size(0)
        motion = voxel_frames.abs().sum(dim=[2, 3, 4])
        motion_normalized = torch.zeros_like(motion)
        for b in range(B):
            max_motion = motion[b].max()
            if max_motion > 0:
                motion_normalized[b] = motion[b] / max_motion
        return motion_normalized

    def forward(self, voxel_grid, use_selector=True, force_full_sequence=False):
        """
        Args:
            force_full_sequence (bool):
                - Phase 1 (Dense): 设为 True。强制使用全量帧。
                - Phase 2 (Sparse): 设为 False。使用 Selector 选帧。
        """
        B, T = voxel_grid.shape[:2]

        # ==========================================
        # 1. 帧选择逻辑 (核心修改)
        # ==========================================
        if force_full_sequence:
            # 🔥 Phase 1: 全量通过 (Dense Mode)
            selected_frames = voxel_grid
            # 伪造 lengths，全是满的
            lengths = torch.full((B,), T, device=voxel_grid.device, dtype=torch.long)
            # 时间间隔均匀分布 0~1
            time_intervals = torch.linspace(0, 1, T, device=voxel_grid.device).unsqueeze(0).repeat(B, 1)
        elif use_selector:
            # 🔥 Phase 2: 稀疏选择 (Sparse Mode)
            selected_frames, lengths, time_intervals = self.selector.select_frames(voxel_grid)
        else:
            # Fallback (一般用不到)
            selected_frames = voxel_grid
            lengths = torch.full((B,), T, device=voxel_grid.device, dtype=torch.long)
            time_intervals = torch.linspace(0, 1, T, device=voxel_grid.device).unsqueeze(0).repeat(B, 1)

        motion_info = self.calculate_motion_info(selected_frames)

        # 2. 编码 (Encoder)
        B, K, C, H, W = selected_frames.shape
        frames_flat = selected_frames.view(B * K, C, H, W)
        encoded_flat = self.encoder(frames_flat)

        _, C_new, H_new, W_new = encoded_flat.shape
        encoded = encoded_flat.view(B, K, C_new, H_new, W_new)

        # 3. LSTM
        lstm_out = self.mt_lstm(encoded, time_intervals, motion_info, lengths)

        # 4. 特征提取 (For Feature Loss)
        final_features = []
        for b in range(B):
            # 注意：取最后一个有效帧的特征
            last_idx = lengths[b] - 1
            feat = lstm_out[b, last_idx]
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten()
            final_features.append(feat)
        student_features_raw = torch.stack(final_features)

        # 5. 对齐和分类
        aligned_features = self.feature_adapter(student_features_raw)
        logits = self.head(lstm_out)  # shape: [B, K_current, 11]

        # ==========================================
        # 6. Logits 补齐 (只在 Phase 2 需要)
        # ==========================================
        if not force_full_sequence:
            max_k = self.selector.max_frames
            curr_k = logits.shape[1]
            if curr_k < max_k:
                padded_logits = torch.zeros(B, max_k, logits.shape[2], device=logits.device)
                padded_logits[:, :curr_k, :] = logits
                logits = padded_logits

        return logits, lengths, aligned_features

    # 🔥 Inference with Early Stopping (早停功能)
    def inference_with_early_stopping(self, voxel_grid, confidence_threshold=0.85):
        """
        推理模式：一旦预测置信度超过阈值，立即停止处理后续帧。
        """
        self.eval()
        with torch.no_grad():
            # 这里简化逻辑：逐步输入帧（伪代码逻辑，实际需要改写 Selector 支持逐步输入）
            # 目前版本我们先直接复用 forward，返回整个序列的预测
            logits, lengths, _ = self.forward(voxel_grid, use_selector=True)
            probs = F.softmax(logits, dim=-1)  # [B, K, Classes]

            # 检查每一步的置信度
            batch_preds = []
            for b in range(logits.size(0)):
                length = lengths[b]
                final_pred = None
                for k in range(length):
                    p = probs[b, k]
                    max_prob, cls = p.max(0)
                    # 如果置信度够高，或者已经到了最后一帧
                    if max_prob > confidence_threshold or k == length - 1:
                        final_pred = (cls.item(), max_prob.item(), k + 1)  # 类别, 置信度, 用了多少帧
                        break
                batch_preds.append(final_pred)
            return batch_preds


# 混合蒸馏 Loss (保持不变)
class HybridDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=10.0, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = temperature

        # 🔥 修改点：加入 label_smoothing=0.1
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, labels, lengths, student_features=None, teacher_features=None):
        # 1. Hard Loss (Classification)
        student_final_logits = []
        for b in range(student_logits.size(0)):
            student_final_logits.append(student_logits[b, lengths[b] - 1])
        student_final = torch.stack(student_final_logits)
        hard_loss = self.ce_loss(student_final, labels)

        # 2. Soft Loss (Logits Distillation)
        soft_loss = 0.0
        if teacher_logits is not None:
            student_soft = F.log_softmax(student_final / self.T, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
            soft_loss = self.kl_div(student_soft, teacher_soft) * (self.T ** 2)

        # 3. Feature Loss
        feat_loss = 0.0
        if self.beta > 0 and student_features is not None and teacher_features is not None:
            feat_loss = self.mse_loss(student_features, teacher_features)

        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss + self.beta * feat_loss
        return total_loss

