import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
import sys
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import random
# 🔥 用于实现 EMA 黑魔法 (向下兼容版)
from torch.optim.swa_utils import AveragedModel

def get_ema_avg_fn(decay=0.999):
    @torch.no_grad()
    def ema_update(ema_param, current_param, num_averaged):
        return decay * ema_param + (1.0 - decay) * current_param
    return ema_update

# ==============================================================================
# 1. 导入模块
# ==============================================================================
try:
    from teacher_model_videomae import TeacherModelVideoMAE
    from student_model_topk_check import StudentModel

    print("✅ 成功导入基础模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(42)

# ==============================================================================
# 2. 📝 SOTA 冲击配置 (DailyAction 特供版)
# ==============================================================================
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'dailyaction_sota_ema_log_{timestamp}.txt'

PARAMS = {
    'resolution': '32x32',
    'student_size': 128,
    'teacher_size': 224,
    'student_hidden': 64,
    'encoder_channels': 64,
    'se_block': True,
    'dropout': 0.3,
    'batch_size': 8,
    'weight_decay': 1e-4,
    'clip_grad': 1.0,

    'epoch_pretrain': 350,
    'lr_pretrain': 5e-3,

    'epoch_finetune': 200,
    'lr_finetune': 5e-3,
    'p2_min_k': 8,
    'p2_max_k': 12,

    'mixup_alpha': 0.03,
    'temp_loss_alpha': 0.5,
    'temp_loss_beta': 0.1,
    'temperature': 3.0,
    'early_exit_threshold': 0.8
}

CONFIG = {
    'data_dir': '/media/amax/US100 2TB/Alxl/Dailyaction_voxel_dataset',
    # 🚨 记得把下面这行换成你真实的 Teacher 模型路径！
    'teacher_weights': './checkpoints/teacher_dailyaction_20260329_113732/teacher_best.pth',
    'teacher_config': {
        'e2vid_path': './rpg_e2vid-master',
        'e2vid_checkpoint': './pretrained_weights/E2VID.pth.tar',
        'videomae_path': './VideoMAEv2-master',
        'videomae_checkpoint': './pretrained_weights/vit_b_k710_dl_from_giant.pth',
        'num_classes': 12, 'lora_rank': 16, 'lora_alpha': 32.0, 'use_lora': True,
        'gray_to_rgb_method': 'conv', 'freeze_weights': True
    },
    'num_classes': 12,
    'batch_size': PARAMS['batch_size'],
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'log_file': log_filename
}


def log_experiment_header():
    lines = [f"🚀 DAILYACTION SOTA DISTILLATION: {timestamp}", "=" * 60]
    for k, v in PARAMS.items(): lines.append(f"  - {k} : {v}")
    lines.append("\n🔥 TRICKS: Asymmetric Distillation + Top-K + EMA + Teacher Masking + SE-Block + Mixup")
    print("\n".join(lines))
    with open(CONFIG['log_file'], "w") as f: f.write("\n".join(lines))


# ==============================================================================
# 3. 工具类 (Augmentation & Dataset)
# ==============================================================================
class VoxelAugmentation:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, voxel_grid):
        T, C, H, W = voxel_grid.shape
        x = voxel_grid.view(-1, H, W)
        if random.random() < 0.5: x = TF.rotate(x, random.uniform(-15, 15))
        if random.random() < 0.5:
            shift = int(H * 0.1)
            x = TF.affine(x, 0, (random.randint(-shift, shift), random.randint(-shift, shift)), 1.0, 0)
        return x.view(T, C, H, W)


class DailyActionVoxelDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            for f in (self.root_dir / cls_name).glob('*.npz'):
                self.samples.append((f, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f, l = self.samples[idx]
        try:
            d = torch.from_numpy(np.load(f)['voxel'].astype(np.float32))
            if self.transform: d = self.transform(d)
            return d, l
        except Exception as e:
            print(f"⚠️ 跳过损坏的样本 {f}: {e}")
            return torch.zeros(20, 5, PARAMS['teacher_size'], PARAMS['teacher_size']), l


def load_robust_teacher_weights(model, weight_path):
    print(f"Loading Teacher: {weight_path} ...")
    checkpoint = torch.load(weight_path, map_location='cpu')
    full_state_dict = {}
    if 'lora_params' in checkpoint: full_state_dict.update(checkpoint['lora_params'])
    if 'recognition_head' in checkpoint:
        for k, v in checkpoint['recognition_head'].items(): full_state_dict[f'recognition_head.{k}'] = v
    if 'gray_to_rgb' in checkpoint:
        for k, v in checkpoint['gray_to_rgb'].items(): full_state_dict[f'gray_to_rgb.{k}'] = v
    res = model.load_state_dict(full_state_dict, strict=False)
    print(f"✅ Teacher weights loaded. Missing keys: {len(res.missing_keys)}")
    return model


def profile_model(model, device):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    msg = f"\n📊 [Profile] Student Total Params: {total_params / 1e6:.4f}M"
    print(msg)
    with open(CONFIG['log_file'], "a") as f:
        f.write(msg + "\n")


# ==============================================================================
# 4. 🔥 极致纯粹的简单 Loss (包含特征形状安全断言)
# ==============================================================================
class HybridDistillationLossSimple(nn.Module):
    def __init__(self, alpha=0.3, beta=10.0, temperature=6.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.T = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='none')
        self.mse_loss = nn.MSELoss()

    def forward(self, student_logits, teacher_logits, labels, lengths, student_features=None, teacher_features=None):
        B = student_logits.size(0)
        batch_indices = torch.arange(B, device=student_logits.device)
        final_student_logits = student_logits[batch_indices, lengths - 1]
        hard_loss = self.ce_loss(final_student_logits, labels)

        soft_loss = 0.0
        if teacher_logits is not None:
            t_preds = teacher_logits.argmax(dim=1)
            correct_mask = (t_preds == labels).float().unsqueeze(1)
            student_soft = F.log_softmax(final_student_logits / self.T, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.T, dim=1)
            kl_loss_per_sample = self.kl_div(student_soft, teacher_soft).sum(dim=1, keepdim=True)
            n_correct = correct_mask.sum().clamp(min=1)
            soft_loss = (kl_loss_per_sample * correct_mask).sum() / n_correct * (self.T ** 2)

        cls_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        feat_loss = 0.0
        if self.beta > 0 and student_features is not None and teacher_features is not None:
            # 🔥 增加特征维度对齐断言
            assert student_features.shape == teacher_features.shape, \
                f"Feature shape mismatch! Student: {student_features.shape}, Teacher: {teacher_features.shape}"
            feat_loss = self.mse_loss(student_features, teacher_features)

        return cls_loss + self.beta * feat_loss


# ==============================================================================
# 5. Trainer
# ==============================================================================
class Trainer:
    def __init__(self, student, teacher, criterion, optimizer, device, writer, scheduler=None, use_ema=False):
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.writer = writer
        self.scheduler = scheduler
        self.global_step = 0
        self.use_ema = use_ema
        if self.use_ema:
            print("🌟 开启 EMA (指数移动平均) 权重平滑技术...")
            self.ema_model = AveragedModel(self.student, avg_fn=get_ema_avg_fn(0.999))
        else:
            self.ema_model = None

    def train_epoch(self, loader, phase_name, epoch, is_phase1):
        self.student.train()
        self.teacher.eval()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(loader, desc=f"{phase_name} Train", leave=False)
        force_full = True if is_phase1 else False

        for voxel_grid, labels in pbar:
            voxel_grid, labels = voxel_grid.to(self.device), labels.to(self.device)

            use_mixup = False
            # 🔥 绝妙设计：Teacher 永远看清晰原图产生纯净目标，Student 挑战混合图拟合！
            if not is_phase1 and PARAMS['mixup_alpha'] > 0 and random.random() < 0.5:
                use_mixup = True
                lam = np.random.beta(PARAMS['mixup_alpha'], PARAMS['mixup_alpha'])
                index = torch.randperm(voxel_grid.size(0)).to(self.device)

                mixed_input = lam * voxel_grid + (1 - lam) * voxel_grid[index, :]
                s_input_224 = mixed_input
                target_a, target_b = labels, labels[index]
            else:
                s_input_224 = voxel_grid
                lam = 1.0

            # ✅ 退格了！必须在 if/else 之外，保证每个 Batch 都雷打不动地执行！
            student_input = s_input_224

            s_logits, lengths, s_feats = self.student(
                voxel_grid=student_input, use_selector=not is_phase1, force_full_sequence=force_full
            )

            with torch.no_grad():
                t_out = self.teacher(voxel_grid)
                t_logits, t_feats = t_out if isinstance(t_out, tuple) else (t_out, None)

            if use_mixup:
                loss_a = self.criterion(s_logits, t_logits, target_a, lengths, s_feats, t_feats)
                loss_b = self.criterion(s_logits,
                                        t_logits[index] if t_logits is not None else None,
                                        target_b, lengths, s_feats,
                                        t_feats[index] if t_feats is not None else None)
                loss = lam * loss_a + (1 - lam) * loss_b
            else:
                loss = self.criterion(s_logits, t_logits, labels, lengths, s_feats, t_feats)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=PARAMS['clip_grad'])
            self.optimizer.step()

            if self.use_ema:
                self.ema_model.update_parameters(self.student)

            if self.writer:
                self.writer.add_scalar(f'{phase_name}/Batch_Loss', loss.item(), self.global_step)
                self.global_step += 1

            total_loss += loss.item()

            # 🔥 高效的向量化计算准确率 (感谢 Claude 的优化建议)
            B = voxel_grid.size(0)
            final_logits = s_logits[torch.arange(B, device=self.device), lengths - 1]
            correct += (final_logits.argmax(dim=1) == labels).sum().item()
            total += B

            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{100 * correct / total:.2f}%"})

        return total_loss / len(loader), 100.0 * correct / total

    def validate(self, loader, phase_name, epoch, is_phase1):
        eval_model = self.ema_model if self.use_ema else self.student
        eval_model.eval()
        self.teacher.eval()  # 🛡️ 安全起见，确保 Teacher 在 Eval 模式

        correct, total, total_loss = 0, 0, 0
        force_full = True if is_phase1 else False

        with torch.no_grad():
            for voxel_grid, labels in tqdm(loader, desc=f"{phase_name} Val", leave=False):
                voxel_grid, labels = voxel_grid.to(self.device), labels.to(self.device)
                # ✅ 改成直接原尺寸传入：
                s_logits, lengths, s_feats = eval_model(voxel_grid=voxel_grid, use_selector=not is_phase1,
                                                        force_full_sequence=force_full)



                teacher_output = self.teacher(voxel_grid)
                t_logits, t_feats = teacher_output if isinstance(teacher_output, tuple) else (teacher_output, None)

                loss = self.criterion(s_logits, t_logits, labels, lengths, s_feats, t_feats)
                total_loss += loss.item()

                # 🔥 向量化加速
                B = voxel_grid.size(0)
                final_logits = s_logits[torch.arange(B, device=self.device), lengths - 1]
                correct += (final_logits.argmax(dim=1) == labels).sum().item()
                total += B

        return total_loss / len(loader), 100.0 * correct / total

    def run_phase(self, train_loader, val_loader, epochs, phase_name, save_dir, is_phase1=True, patience=None):
        os.makedirs(save_dir, exist_ok=True)
        best_acc = 0.0
        epochs_no_improve = 0
        print(f"\n=== START {phase_name} ({epochs} Epochs) [Patience={patience}] ===")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, phase_name, epoch, is_phase1)

            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]

            val_loss, val_acc = self.validate(val_loader, phase_name, epoch, is_phase1)

            if self.writer:
                self.writer.add_scalar(f'{phase_name}/Loss/Train', train_loss, epoch)
                self.writer.add_scalar(f'{phase_name}/Loss/Val', val_loss, epoch)
                self.writer.add_scalar(f'{phase_name}/Acc/Val', val_acc, epoch)
                self.writer.add_scalar(f'{phase_name}/LR', current_lr, epoch)

            log_str = (f"Epoch [{epoch + 1}/{epochs}] LR={current_lr:.6f} "
                       f"Train: Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
                       f"Val: Loss={val_loss:.4f} Acc={val_acc:.2f}%")
            print(log_str)
            with open(CONFIG['log_file'], "a") as f:
                f.write(log_str + "\n")

            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0

                if self.use_ema:
                    base_model = self.ema_model.module
                    save_dict = base_model.module.state_dict() if hasattr(base_model,
                                                                          'module') else base_model.state_dict()
                else:
                    save_dict = self.student.module.state_dict() if hasattr(self.student,
                                                                            'module') else self.student.state_dict()

                torch.save({'model_state_dict': save_dict}, os.path.join(save_dir, f"student_{phase_name}_best.pth"))
                save_msg = f"  ✓ Saved Best (Acc: {best_acc:.2f}%)"
                print(save_msg)
                with open(CONFIG['log_file'], "a") as f:
                    f.write(save_msg + "\n")
            else:
                epochs_no_improve += 1

            if patience is not None and epochs_no_improve >= patience:
                stop_msg = f"\n🛑 Early Stopping! No improvement for {patience} epochs. Stopping {phase_name}."
                print(stop_msg)
                with open(CONFIG['log_file'], "a") as f: f.write(stop_msg + "\n")
                break

        return best_acc


# ==============================================================================
# 6. 早停测试 (多阈值扫描器版)
# ==============================================================================
def run_early_exit_test(student, val_loader, device, log_file):
    header = f"\n" + "=" * 50 + "\n▶ FINAL TEST: EARLY EXIT TRADE-OFF\n" + "=" * 50
    print(header)
    with open(log_file, "a") as f:
        f.write(header + "\n")

    student.eval()
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    results = {th: {'correct': 0, 'total_frames': 0} for th in thresholds}
    total_samples = 0

    with torch.no_grad():
        for voxel_grid, labels in tqdm(val_loader, desc="Scanning Thresholds"):
            voxel_grid, labels = voxel_grid.to(device), labels.to(device)
            B, T, C, H, W = voxel_grid.shape

            # 🚨 保持与训练阶段一致，直接输入 224x224 原图！
            student_input = voxel_grid

            total_samples += B
            logits, lengths, _ = student(student_input, use_selector=True, force_full_sequence=False)
            probs = F.softmax(logits, dim=-1)

            for b in range(B):
                T_curr = lengths[b].item()
                for th in thresholds:
                    predicted = False
                    for t in range(T_curr):
                        p_max, cls = probs[b, t].max(0)
                        if p_max.item() >= th or t == T_curr - 1:
                            if cls.item() == labels[b].item():
                                results[th]['correct'] += 1
                            results[th]['total_frames'] += (t + 1)
                            predicted = True
                            break
                    if not predicted:
                        results[th]['total_frames'] += T_curr

    table_str = "\n📊 [Early Exit Trade-off Table for Paper]\n"
    table_str += f"{'Threshold':<12} | {'Accuracy (%)':<15} | {'Avg Frames':<12}\n"
    table_str += "-" * 45 + "\n"
    for th in thresholds:
        acc = 100.0 * results[th]['correct'] / total_samples
        avg_f = results[th]['total_frames'] / total_samples
        table_str += f"{th:<12.2f} | {acc:<15.2f} | {avg_f:<12.2f}\n"
    table_str += "=" * 50 + "\n"

    print(table_str)
    with open(log_file, "a") as f:
        f.write(table_str)


# ==============================================================================
# 7. 主程序
# ==============================================================================
def main():
    # 🚨 Dummy Path Guard (防御假路径导致瞬间崩塌)
    if 'xxxxxxxx' in CONFIG['teacher_weights']:
        raise ValueError("❌ 停！请先把 CONFIG 里的 teacher_weights 改成真实的模型路径再跑！")
    if not Path(CONFIG['teacher_weights']).exists():
        raise FileNotFoundError(f"❌ 找不到 Teacher 权重: {CONFIG['teacher_weights']}")

    log_experiment_header()
    device = CONFIG['device']
    writer = SummaryWriter(log_dir=f'runs/{timestamp}_dailyaction_sota_ema')

    print("[1/5] Loading Data...")
    augmentor = VoxelAugmentation(size=PARAMS['teacher_size'])  # 必须生成 224 给老师
    train_set = DailyActionVoxelDataset(os.path.join(CONFIG['data_dir'], 'train'), transform=augmentor)
    val_set = DailyActionVoxelDataset(os.path.join(CONFIG['data_dir'], 'test'), transform=None)

    # 🛡️ 数据集类目校验
    actual_classes = len(train_set.classes)
    assert actual_classes == CONFIG[
        'num_classes'], f"❌ Dataset has {actual_classes} classes but CONFIG says {CONFIG['num_classes']}"
    print(f"✅ Class count verified: {actual_classes} — {train_set.classes}")

    train_loader = DataLoader(
        train_set, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=8, pin_memory=True
    )

    print("[2/5] Loading Teacher...")
    teacher = TeacherModelVideoMAE(**CONFIG['teacher_config'], device=device)
    teacher = load_robust_teacher_weights(teacher, CONFIG['teacher_weights'])
    teacher.to(device).eval()

    if torch.cuda.device_count() > 1:
        teacher = nn.DataParallel(teacher)

    # =========================================================
    # PHASE 1: Dense Pre-training
    # =========================================================
    print("\n▶ PHASE 1 START (Dense - 20 Frames)")
    student = StudentModel(
        encoder_channels=PARAMS['encoder_channels'],
        lstm_hidden=PARAMS['student_hidden'],
        lstm_layers=1, num_classes=CONFIG['num_classes'],
        min_keyframes=20, max_keyframes=20,
        dropout=PARAMS['dropout'],
        diff_threshold=0.5
    ).to(device)

    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)

    profile_model(student, device)

    opt1 = optim.Adam(student.parameters(), lr=PARAMS['lr_pretrain'], weight_decay=PARAMS['weight_decay'])
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=PARAMS['epoch_pretrain'], eta_min=5e-6)
    crit1 = HybridDistillationLossSimple(alpha=0.5, beta=PARAMS['temp_loss_beta'], temperature=PARAMS['temperature'])

    trainer1 = Trainer(student, teacher, crit1, opt1, device, writer, sched1)
    save_dir_name = f"checkpoints/dailyaction_student_{timestamp}"

    trainer1.run_phase(train_loader, val_loader, PARAMS['epoch_pretrain'], 'Phase1', save_dir_name, is_phase1=True,
                       patience=None)

    # =========================================================
    # PHASE 2: Sparse Fine-tuning (稀疏跳帧)
    # =========================================================
    print("\n▶ PHASE 2 START (Sparse Top-K)")
    ft_student = StudentModel(
        encoder_channels=PARAMS['encoder_channels'], lstm_hidden=PARAMS['student_hidden'],
        lstm_layers=1, num_classes=CONFIG['num_classes'], min_keyframes=PARAMS['p2_min_k'],
        max_keyframes=PARAMS['p2_max_k'],
        dropout=PARAMS['dropout'],
        diff_threshold=0.5
    ).to(device)

    ckpt_path = f"{save_dir_name}/student_Phase1_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['model_state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        ft_student.load_state_dict(new_state_dict)
    else:
        print(f"⚠️ 警告: 未找到 Phase 1 权重 {ckpt_path}，将从头开始 Phase 2！")

    if torch.cuda.device_count() > 1:
        ft_student = nn.DataParallel(ft_student)

    opt2 = optim.Adam(ft_student.parameters(), lr=PARAMS['lr_finetune'], weight_decay=PARAMS['weight_decay'])

    # 🚨 增加 Phase 2 的退火调度器 (Claude 强烈建议，防止局部震荡)
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=PARAMS['epoch_finetune'], eta_min=1e-7)

    crit2 = HybridDistillationLossSimple(alpha=PARAMS['temp_loss_alpha'], beta=PARAMS['temp_loss_beta'],
                                         temperature=PARAMS['temperature'])

    trainer2 = Trainer(ft_student, teacher, crit2, opt2, device, writer, scheduler=sched2, use_ema=True)
    trainer2.run_phase(train_loader, val_loader, epochs=PARAMS['epoch_finetune'], phase_name='Phase2',
                       save_dir=save_dir_name, is_phase1=False, patience=200)

    # =========================================================
    # FINAL TEST: Early Exit Trade-off
    # =========================================================
    best_ckpt = torch.load(f"{save_dir_name}/student_Phase2_best.pth", map_location=device)
    test_state_dict = {k.replace('module.', ''): v for k, v in best_ckpt['model_state_dict'].items()}

    if isinstance(ft_student, nn.DataParallel):
        ft_student.module.load_state_dict(test_state_dict)
    else:
        ft_student.load_state_dict(test_state_dict)

    run_early_exit_test(ft_student, val_loader, device, CONFIG['log_file'])
    print(f"\n✅ All Done! Log: {CONFIG['log_file']}")
    writer.close()


if __name__ == "__main__":
    main()