import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import sys
import os
import math


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning.
    Adds trainable low-rank matrices A and B to frozen weight matrix W.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA forward: x @ (A @ B) * scaling
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Combines frozen pretrained weights with trainable low-rank updates.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

        # Freeze original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False

        # 保存原始权重用于 forward
        self.register_buffer('merged_weight', self.linear.weight.data.clone())

    @property
    def weight(self):
        # 返回合并后的权重，VideoMAE 调用 F.linear 时可用
        return self.merged_weight + self.lora.scaling * (self.lora.lora_B.T @ self.lora.lora_A.T)

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, weight=self.weight, bias=self.bias)


class GrayscaleToRGBConverter(nn.Module):
    """
    Convert grayscale images to RGB using either channel copy or 1x1 convolution.
    """

    def __init__(self, method: str = 'copy'):
        """
        Args:
            method: 'copy' (repeat channels) or 'conv' (learnable 1x1 conv)
        """
        super().__init__()
        self.method = method

        if method == 'conv':
            # Learnable 1x1 convolution to convert 1 channel to 3 channels
            self.conv = nn.Conv2d(1, 3, kernel_size=1, bias=True)
            # Initialize to approximate channel copying
            with torch.no_grad():
                self.conv.weight.fill_(1.0)
                self.conv.bias.zero_()
        elif method != 'copy':
            raise ValueError(f"Invalid method: {method}. Choose 'copy' or 'conv'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Grayscale images [B, 1, H, W] or [B, 1, T, H, W]
        Returns:
            RGB images [B, 3, H, W] or [B, 3, T, H, W]
        """
        if self.method == 'copy':
            # Simple channel repetition
            return x.repeat(1, 3, *([1] * (x.dim() - 2)))
        else:  # conv
            if x.dim() == 5:  # [B, 1, T, H, W]
                B, C, T, H, W = x.shape
                # Reshape to [B*T, 1, H, W] for conv2d
                x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                x = self.conv(x)
                # Reshape back to [B, 3, T, H, W]
                x = x.reshape(B, T, 3, H, W).permute(0, 2, 1, 3, 4)
            else:  # [B, 1, H, W]
                x = self.conv(x)
            return x


class TeacherModelVideoMAE(nn.Module):
    """
    Teacher model for DVS-Gesture recognition using VideoMAE V2 with LoRA.
    Pipeline: Voxel Grid Events -> E2VID -> RGB Video -> VideoMAE V2 + LoRA -> Recognition Head
    """

    def __init__(
            self,
            e2vid_path: str,
            e2vid_checkpoint: str,
            videomae_path: str,
            videomae_checkpoint: str,
            num_classes: int = 11,
            lora_rank: int = 8,
            lora_alpha: float = 16.0,
            use_lora: bool = True,
            gray_to_rgb_method: str = 'copy',  # 'copy' or 'conv'
            freeze_weights: bool = True,
            device: str = 'cuda'
    ):
        """
        Args:
            e2vid_path: Path to E2VID repository
            e2vid_checkpoint: Path to E2VID checkpoint
            videomae_path: Path to VideoMAE V2 repository
            videomae_checkpoint: Path to VideoMAE V2 checkpoint
            num_classes: Number of classes (11 for DVS-Gesture)
            lora_rank: Rank of LoRA matrices
            lora_alpha: LoRA scaling factor
            use_lora: Whether to use LoRA adaptation
            gray_to_rgb_method: 'copy' (channel repetition) or 'conv' (1x1 conv)
            freeze_weights: Whether to freeze backbone weights
            device: Device to run on
        """
        super(TeacherModelVideoMAE, self).__init__()

        self.device = device
        self.num_classes = num_classes
        self.use_lora = use_lora
        self.num_frames = 16  # VideoMAE V2 standard input

        # Add paths to system
        if e2vid_path not in sys.path:
            sys.path.insert(0, e2vid_path)
        if videomae_path and videomae_path not in sys.path:
            sys.path.insert(0, videomae_path)

        # Load E2VID model
        self.e2vid = self._load_e2vid(e2vid_checkpoint)

        # Grayscale to RGB converter
        self.gray_to_rgb = GrayscaleToRGBConverter(method=gray_to_rgb_method)
        print(f"✓ Using grayscale-to-RGB method: {gray_to_rgb_method}")

        # Load VideoMAE V2 backbone
        self.videomae_backbone = self._load_videomae(videomae_path, videomae_checkpoint)

        # VideoMAE ViT-B embedding dimension
        self.videomae_embed_dim = 768

        # Replace original head with our recognition head
        if hasattr(self.videomae_backbone, 'head'):
            # Remove the original head (Linear(768, 710))
            self.videomae_backbone.head = nn.Identity()
            print(f"✓ Removed original VideoMAE head (Linear(768, 710))")

        # Recognition head for DVS-Gesture
        self.recognition_head = nn.Sequential(
            nn.LayerNorm(self.videomae_embed_dim),
            nn.Dropout(0.5),
            nn.Linear(self.videomae_embed_dim, num_classes)
        ).to(self.device)
        print(f"✓ Created recognition head: Linear({self.videomae_embed_dim}, {num_classes})")

        # Apply LoRA if enabled
        if use_lora and freeze_weights:
            self._apply_lora(lora_rank, lora_alpha)

        # Freeze backbone weights if specified
        if freeze_weights:
            self._freeze_backbone_weights()

        self.to(self.device)

    def _load_e2vid(self, checkpoint_path: str) -> nn.Module:
        """Load pre-trained E2VID model."""
        try:
            print(f"Loading E2VID checkpoint from: {checkpoint_path}")

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if 'arch' in checkpoint:
                print(f"Architecture: {checkpoint['arch']}")

            # Extract state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint and isinstance(checkpoint['model'], dict):
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Check for wrapper
            has_wrapper = any(k.startswith('unetrecurrent.') for k in state_dict.keys())

            from model.model import E2VIDRecurrent

            if has_wrapper:
                config = {
                    'num_bins': 5,
                    'num_output_channels': 1,
                    'skip_type': 'sum',
                    'recurrent_block_type': 'convlstm',
                    'activation': 'relu',
                    'num_encoders': 3,
                    'base_num_channels': 32,
                    'num_residual_blocks': 2,
                    'norm': 'BN',
                    'use_upsample_conv': True
                }
                model = E2VIDRecurrent(config)
            else:
                from model.model import UNetRecurrent
                model = UNetRecurrent(
                    num_input_channels=5,
                    num_output_channels=1,
                    skip_type='sum',
                    recurrent_block_type='convlstm',
                    activation='relu',
                    num_encoders=3,
                    base_num_channels=32,
                    num_residual_blocks=2,
                    norm='BN',
                    use_upsample_conv=True
                )

            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()

            print("✓ E2VID model loaded successfully")
            return model

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load E2VID: {str(e)}")

    def _load_videomae(self, videomae_path: str, checkpoint_path: str) -> nn.Module:
        print(f"Loading VideoMAE V2 from: {videomae_path}")

        if videomae_path not in sys.path:
            sys.path.insert(0, videomae_path)

        from models import modeling_finetune

        # 初始化模型 (默认 head=1000类)
        model = modeling_finetune.vit_base_patch16_224(
            pretrained=False,
            all_frames=16,
            tubelet_size=2,
            use_mean_pooling=True
        )

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 1. 拆包逻辑 (保留之前的修复)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'module' in checkpoint and isinstance(checkpoint['module'], dict):
            print("  Found nested 'module' key, unpacking...")
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint

        # 2. 键名清洗 + 形状不匹配剔除
        new_state_dict = {}
        for k, v in state_dict.items():
            # 清洗前缀
            if k.startswith("module."): k = k[7:]
            if k.startswith("encoder."): k = k[8:]
            if k.startswith("backbone."): k = k[9:]

            # === 新增修复：直接剔除 head 层 ===
            # 因为 710类(权重) vs 1000类(模型) 会导致 size mismatch 报错
            # 我们反正后面要换成 11类，所以这里直接不加载 head
            if k.startswith("head."):
                continue

            new_state_dict[k] = v

        # 调试打印
        model_keys = list(model.state_dict().keys())
        ckpt_keys = list(new_state_dict.keys())
        if len(ckpt_keys) > 0:
            print(f"  Example model key: {model_keys[0]}")
            print(f"  Example ckpt key:  {ckpt_keys[0]}")

        # 加载权重
        msg = model.load_state_dict(new_state_dict, strict=False)

        print("✓ VideoMAE weights loaded")
        # 这里 Missing keys 肯定会包含 head.weight 等，这是符合预期的
        print(f"  Missing keys: {len(msg.missing_keys)}")

        # 再次确认：只要 backbone 没丢就行
        # 排除掉 head 相关的缺失键后，剩下的缺失键应该几乎为 0
        real_missing = [k for k in msg.missing_keys if not k.startswith("head.")]
        if len(real_missing) > 10:
            print("\n❌ CRITICAL ERROR: Backbone still empty!")
            print(f"Top 5 missing: {real_missing[:5]}")
            raise RuntimeError("VideoMAE backbone load failed.")

        model.to(self.device)
        model.eval()
        return model

    def _apply_lora(self, rank: int, alpha: float):
        """Apply LoRA to all attention layers in VideoMAE backbone."""
        print(f"\nApplying LoRA (rank={rank}, alpha={alpha})...")

        lora_count = 0
        for name, module in self.videomae_backbone.named_modules():
            # Apply LoRA to attention projection layers (qkv and proj)
            if isinstance(module, nn.Linear):
                # Target attention layers (qkv and output projection)
                if 'attn.qkv' in name or 'attn.proj' in name:
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]

                    parent = self.videomae_backbone
                    for part in parent_name.split('.'):
                        if part:
                            parent = getattr(parent, part)

                    # Replace with LoRA layer
                    lora_linear = LoRALinear(module, rank=rank, alpha=alpha)
                    setattr(parent, child_name, lora_linear)
                    lora_count += 1

        print(f"✓ Applied LoRA to {lora_count} layers")

    def _freeze_backbone_weights(self):
        """Freeze all backbone parameters except LoRA and recognition head."""
        # Freeze E2VID
        for param in self.e2vid.parameters():
            param.requires_grad = False

        # Freeze VideoMAE backbone (except LoRA if present)
        for name, param in self.videomae_backbone.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

        # Freeze grayscale converter if using copy method
        if self.gray_to_rgb.method == 'copy':
            for param in self.gray_to_rgb.parameters():
                param.requires_grad = False

        print("✓ Backbone weights frozen (LoRA and recognition head remain trainable)")

    def reconstruct_images(self, voxel_grid: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct grayscale images from event voxel grids using E2VID.
        [修复版]：支持 Batch 并行处理，修复多卡训练报错。
        Args:
            voxel_grid: [B, T, C, H, W] where T=20, C=5, H=W=128
        Returns:
            Grayscale images [B, 1, T, H, W]
        """
        B, T, C, H, W = voxel_grid.shape

        # Process frame by frame (Time Loop only)
        # E2VID (UNetRecurrent) 天然支持 Batch 输入 [B, C, H, W]
        with torch.no_grad():
            if hasattr(self.e2vid, 'unetrecurrent'):
                reconstructed_frames = []
                prev_states = None  # 初始化状态

                for t in range(T):
                    # 取出当前时刻所有样本: [B, C, H, W]
                    frames_batch = voxel_grid[:, t, :, :, :]

                    # 🔥 直接整批送入，不需要 for b 循环！
                    # output 可能是 dict 或 tuple，取决于具体实现
                    output = self.e2vid(frames_batch, prev_states)

                    if isinstance(output, dict):
                        img = output['image']  # [B, 1, H, W]
                        prev_states = output.get('states', None)
                    elif isinstance(output, tuple):
                        img, prev_states = output
                    else:
                        img = output
                        prev_states = None

                    reconstructed_frames.append(img)

                # 将时间维度堆叠回来: List[B, 1, H, W] -> [B, T, 1, H, W]
                # 注意 stack 的维度是 1 (时间维)
                reconstructed_gray = torch.stack(reconstructed_frames, dim=1)

            else:
                # 非递归模式 (很少见，但也保留逻辑)
                voxel_flat = voxel_grid.reshape(B * T, C, H, W)
                reconstructed_gray = self.e2vid(voxel_flat)
                if isinstance(reconstructed_gray, dict):
                    reconstructed_gray = reconstructed_gray['image']
                reconstructed_gray = reconstructed_gray.reshape(B, T, 1, H, W)

            # Ensure single channel [B, T, 1, H, W] -> [B, 1, T, H, W]
            if reconstructed_gray.dim() == 4:
                reconstructed_gray = reconstructed_gray.unsqueeze(2)

            # Normalize to [0, 1]
            if reconstructed_gray.min() < 0 or reconstructed_gray.max() > 1:
                reconstructed_gray = torch.sigmoid(reconstructed_gray)

            # Permute to [B, 1, T, H, W] for downstream
            reconstructed_gray = reconstructed_gray.permute(0, 2, 1, 3, 4)

        return reconstructed_gray

    def _preprocess_for_videomae(self, images: torch.Tensor) -> torch.Tensor:
        """
        Preprocess grayscale images for VideoMAE V2 input.

        Args:
            images: Grayscale [B, 1, T, H, W] where T=20, H=W=128

        Returns:
            RGB video [B, 3, num_frames, 224, 224] where num_frames=16
        """
        B, C, T, H, W = images.shape

        # Step 1: Uniform temporal sampling from 20 frames to 16 frames
        if T != self.num_frames:
            # Uniform sampling: evenly spaced indices
            indices = torch.linspace(0, T - 1, self.num_frames).long()
            images = images[:, :, indices, :, :]  # [B, 1, 16, 128, 128]

        # Step 2: Convert grayscale to RGB (1 -> 3 channels)
        images = self.gray_to_rgb(images)  # [B, 3, 16, 128, 128]

        # Step 3: Resize from 128x128 to 224x224 using bilinear interpolation
        B, C, T, H, W = images.shape
        images = images.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = images.reshape(B, T, C, 224, 224).permute(0, 2, 1, 3, 4)  # [B, 3, 16, 224, 224]

        # Step 4: Normalize with ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(images.device)
        images = (images - mean) / std

        return images


    def extract_features(self, video: torch.Tensor) -> torch.Tensor:
        """
                Extract features from RGB video using VideoMAE V2.

                Args:
                    video: [B, 3, T, 224, 224]

                Returns:
                    Features [B, 768]
                """
        # VideoMAE backbone (already has head removed)
        features = self.videomae_backbone.forward_features(video)
        return features

    def forward(self, voxel_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            voxel_grid: [B, T, C, H, W] = [B, 20, 5, 128, 128]

        Returns:
            logits: [B, num_classes]
            features: [B, 768]
        """
        # Step 1: E2VID reconstruction -> [B, 1, 20, 128, 128]
        reconstructed_gray = self.reconstruct_images(voxel_grid)

        # Step 2: Preprocess -> [B, 3, 16, 224, 224]
        video_tensor = self._preprocess_for_videomae(reconstructed_gray)

        # Step 3: Extract features -> [B, 768]
        features = self.extract_features(video_tensor)

        # Step 4: Recognition head -> [B, num_classes]
        logits = self.recognition_head(features)

        return logits, features

    def get_soft_targets(self, voxel_grid: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Get soft targets for knowledge distillation."""
        logits, _ = self.forward(voxel_grid)
        soft_targets = F.softmax(logits / temperature, dim=1)
        return soft_targets


# Knowledge Distillation Loss
class KnowledgeDistillationLoss(nn.Module):
    """Combined loss for knowledge distillation."""

    def __init__(self, alpha: float = 0.5, temperature: float = 3.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Hard target loss
        hard_loss = self.ce_loss(student_logits, targets)

        # Soft target loss
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.temperature ** 2)

        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss

        return total_loss, distillation_loss, hard_loss


if __name__ == "__main__":
    """Example usage"""

    # Test with channel copy method
    print("\n" + "=" * 60)
    print("Testing with channel copy method")
    print("=" * 60)
    teacher_copy = TeacherModelVideoMAE(
        e2vid_path='./rpg_e2vid-master',
        e2vid_checkpoint='./pretrained_weights/E2VID.pth.tar',
        videomae_path='./VideoMAEv2-master',
        videomae_checkpoint='./pretrained_weights/vit_b_k710_dl_from_giant.pth',
        num_classes=11,
        lora_rank=8,
        lora_alpha=16.0,
        use_lora=True,
        gray_to_rgb_method='copy',
        freeze_weights=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(2, 20, 5, 128, 128).to(device)

    print("\nForward pass test...")
    logits, features = teacher_copy(dummy_input)
    print(f"✓ Input: {dummy_input.shape}")
    print(f"✓ Logits: {logits.shape}")
    print(f"✓ Features: {features.shape}")

    # Count trainable parameters
    total_params = sum(p.numel() for p in teacher_copy.parameters())
    trainable_params = sum(p.numel() for p in teacher_copy.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")