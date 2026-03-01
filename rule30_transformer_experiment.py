"""
Python 3.11
Rule 30 + PyTorch Transformer demo:
1) Token size is configurable in cells (`token_cells`).
2) Each training sample is a full trajectory of rows (temporal context).
3) Model predicts next row autoregressively with access to previous rows.
4) UI shows exact vs predicted rollout.
"""

from __future__ import annotations

import argparse
import importlib
import random
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk


def rule30_next_row(row: torch.Tensor) -> torch.Tensor:
    left = torch.zeros_like(row)
    left[1:] = row[:-1]
    center = row
    right = torch.zeros_like(row)
    right[:-1] = row[1:]
    pattern = (left << 2) | (center << 1) | right
    return ((30 >> pattern) & 1).to(torch.long)


def rollout_rule30(initial: torch.Tensor, steps: int, extra_margin: int = 0) -> torch.Tensor:
    # Simulate on expanded width, then crop center window each step.
    # This avoids artificial boundary artifacts from fixed-width simulation.
    width = initial.numel()
    growth_margin = max(0, steps - 1)
    margin = growth_margin + max(0, extra_margin)
    full_width = width + 2 * margin

    cur_full = torch.zeros(full_width, dtype=torch.long)
    start = margin
    end = start + width
    cur_full[start:end] = initial.clone().long()

    out = [cur_full[start:end].clone()]
    for _ in range(steps - 1):
        cur_full = rule30_next_row(cur_full)
        out.append(cur_full[start:end].clone())
    return torch.stack(out, dim=0)


@dataclass
class Config:
    width: int = 64
    steps: int = 128
    sim_extra_margin: int = 0
    token_cells: int = 3
    train_sequences: int = 20480
    val_sequences: int = 16
    batch_size: int = 16
    epochs: int = 15
    lr: float = 1e-4
    d_model: int = 512
    nhead: int = 8
    layers: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    edge_ignore: int = 32
    seed: int = 42
    cell_size: int = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule 30 transformer experiment")
    parser.add_argument("--weights-in", type=str, default=None, help="Path to load model weights (.pt/.pth)")
    parser.add_argument("--weights-out", type=str, default=None, help="Path to save model weights (.pt/.pth)")
    parser.add_argument("--inference-only", action="store_true", help="Run inference/evaluation only (skip training)")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile and run eager mode")
    parser.add_argument(
        "--inference-total-steps",
        type=int,
        default=None,
        help="Total rollout steps for inference benchmark; last cfg.steps rows are shown/evaluated",
    )
    parser.add_argument(
        "--inference-block-steps",
        type=int,
        default=32,
        help="Neural rollout block size while keeping sliding temporal context",
    )
    return parser.parse_args()


class TemporalTokenTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if cfg.token_cells <= 0:
            raise ValueError("token_cells must be >= 1")

        self.width = cfg.width
        self.steps = cfg.steps
        self.token_cells = cfg.token_cells
        self.ntok = (cfg.width + cfg.token_cells - 1) // cfg.token_cells
        self.padded_width = self.ntok * cfg.token_cells

        self.token_proj = nn.Linear(cfg.token_cells, cfg.d_model)
        self.spatial_pos = nn.Embedding(self.ntok, cfg.d_model)
        self.time_pos = nn.Embedding(cfg.steps - 1, cfg.d_model)

        spatial_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.spatial_encoder = nn.TransformerEncoder(spatial_layer, num_layers=2)

        temporal_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=cfg.layers)
        self.head = nn.Linear(cfg.d_model, self.padded_width)

        self.register_buffer("spatial_ids", torch.arange(self.ntok).long(), persistent=False)
        self.register_buffer("time_ids", torch.arange(cfg.steps - 1).long(), persistent=False)

    def _pad_width(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, W]
        pad = self.padded_width - x.shape[-1]
        if pad <= 0:
            return x
        return F.pad(x, (0, pad), value=0.0)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, W] -> [B, T, ntok, token_cells]
        x = self._pad_width(x)
        b, t, _ = x.shape
        return x.view(b, t, self.ntok, self.token_cells)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, W], values in {0,1}
        b, t, w = x.shape
        if w != self.width:
            raise ValueError(f"expected width={self.width}, got {w}")
        if t > self.steps - 1:
            raise ValueError(f"time length {t} exceeds max supported {self.steps - 1}")

        x = x * 2.0 - 1.0
        tok = self._tokenize(x)  # [B, T, ntok, token_cells]
        tok = tok.view(b * t, self.ntok, self.token_cells)
        h = self.token_proj(tok) + self.spatial_pos(self.spatial_ids).unsqueeze(0)
        h = self.spatial_encoder(h)
        row_state = h.mean(dim=1).view(b, t, -1) + self.time_pos(self.time_ids[:t]).unsqueeze(0)

        # Predict next row at each timestep without looking into future rows.
        causal_mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        z = self.temporal_encoder(row_state, mask=causal_mask)
        out = self.head(z)[..., : self.width]  # [B, T, W]
        return out


def make_dataset(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    # x: [N, steps-1, W], y: [N, steps-1, W]
    n = cfg.train_sequences
    t = cfg.steps - 1
    w = cfg.width
    dataset_seed = random.SystemRandom().randrange(0, 2**63 - 1)
    dataset_gen = torch.Generator(device="cpu")
    dataset_gen.manual_seed(dataset_seed)
    print(f"dataset seed: {dataset_seed}")
    x = torch.empty((n, t, w), dtype=torch.uint8)
    y = torch.empty((n, t, w), dtype=torch.uint8)
    for i in range(n):
        init = torch.randint(0, 2, (w,), dtype=torch.long, generator=dataset_gen)
        traj = rollout_rule30(init, cfg.steps, extra_margin=cfg.sim_extra_margin).to(torch.uint8)
        x[i] = traj[:-1]
        y[i] = traj[1:]
    return x, y


def make_rollout_validation_set(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    # val_init: [N, W], val_truth: [N, steps, W]
    n = cfg.val_sequences
    w = cfg.width
    s = cfg.steps
    val_seed = random.SystemRandom().randrange(0, 2**63 - 1)
    val_gen = torch.Generator(device="cpu")
    val_gen.manual_seed(val_seed)
    print(f"rollout val seed: {val_seed}")

    val_init = torch.empty((n, w), dtype=torch.long)
    val_truth = torch.empty((n, s, w), dtype=torch.long)
    for i in range(n):
        init = torch.randint(0, 2, (w,), dtype=torch.long, generator=val_gen)
        val_init[i] = init
        val_truth[i] = rollout_rule30_last_window(
            init, total_steps=s, window_steps=s, extra_margin=cfg.sim_extra_margin
        )
    return val_init, val_truth


def _slice_center(t: torch.Tensor, edge_ignore: int) -> torch.Tensor:
    if edge_ignore <= 0:
        return t
    width = t.shape[-1]
    if edge_ignore * 2 >= width:
        return t
    return t[..., edge_ignore : width - edge_ignore]


@torch.no_grad()
def evaluate_free_rollout_set(
    model: nn.Module,
    val_init: torch.Tensor,
    val_truth: torch.Tensor,
    cfg: Config,
    device: torch.device,
    block_steps: int,
) -> tuple[float, float]:
    model.eval()
    total_bits = 0
    total_correct = 0
    total_center_bits = 0
    total_center_correct = 0

    for i in range(val_init.size(0)):
        pred = rollout_model_last_window_blocked(
            model,
            val_init[i],
            total_steps=cfg.steps,
            context_steps=cfg.steps,
            block_steps=block_steps,
            device=device,
        )
        truth = val_truth[i]
        total_correct += (pred == truth).sum().item()
        total_bits += truth.numel()

        pred_center = _slice_center(pred, cfg.edge_ignore)
        truth_center = _slice_center(truth, cfg.edge_ignore)
        total_center_correct += (pred_center == truth_center).sum().item()
        total_center_bits += truth_center.numel()

    return total_correct / max(total_bits, 1), total_center_correct / max(total_center_bits, 1)


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cfg: Config,
    device: torch.device,
    val_init: torch.Tensor,
    val_truth: torch.Tensor,
    rollout_block_steps: int,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    x = x.pin_memory()
    y = y.pin_memory()
    n = x.size(0)
    indices = torch.arange(n)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"mixed precision: {amp_dtype} (scaler={use_scaler})")

    for epoch in range(1, cfg.epochs + 1):
        start_epoch = time.perf_counter()
        indices = indices[torch.randperm(n)]
        total_loss = 0.0
        total_bits = 0
        total_correct = 0
        total_center_bits = 0
        total_center_correct = 0
        total_base_correct = 0
        total_base_center_correct = 0
        steps_count = 0

        for i in range(0, n, cfg.batch_size):
            batch_idx = indices[i : i + cfg.batch_size]
            xb = x[batch_idx].to(device, dtype=torch.float32, non_blocking=True)
            yb = y[batch_idx].to(device, dtype=torch.float32, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(xb)
                loss = F.binary_cross_entropy_with_logits(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            pred = (logits > 0).to(torch.float32)
            total_loss += loss.item() * xb.size(0)
            total_correct += (pred == yb).sum().item()
            total_bits += yb.numel()

            pred_center = _slice_center(pred, cfg.edge_ignore)
            yb_center = _slice_center(yb, cfg.edge_ignore)
            total_center_correct += (pred_center == yb_center).sum().item()
            total_center_bits += yb_center.numel()

            total_base_correct += (yb == 0).sum().item()
            total_base_center_correct += (yb_center == 0).sum().item()
            steps_count += 1

        elapsed = time.perf_counter() - start_epoch
        sec_per_step = elapsed / max(steps_count, 1)
        rollout_val_acc, rollout_val_center_acc = evaluate_free_rollout_set(
            model,
            val_init=val_init,
            val_truth=val_truth,
            cfg=cfg,
            device=device,
            block_steps=rollout_block_steps,
        )
        model.train()
        print(
            f"epoch {epoch:02d} | loss={total_loss / n:.4f} "
            f"| bit_acc={total_correct / total_bits:.4f} "
            f"| center_acc={total_center_correct / max(total_center_bits, 1):.4f} "
            f"| base0_acc={total_base_correct / total_bits:.4f} "
            f"| base0_center={total_base_center_correct / max(total_center_bits, 1):.4f} "
            f"| rollout_val_acc={rollout_val_acc:.4f} "
            f"| rollout_val_center={rollout_val_center_acc:.4f} "
            f"| sec/step={sec_per_step:.3f} | epoch_sec={elapsed:.1f}",
            flush=True,
        )


@torch.no_grad()
def rollout_model(model: nn.Module, initial: torch.Tensor, steps: int, device: torch.device) -> torch.Tensor:
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cur = initial.clone().to(device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, W]
    out = [initial.clone().long().cpu()]

    for _ in range(steps - 1):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(cur)
        next_row = (logits[:, -1, :] > 0).to(torch.float32).unsqueeze(1)  # [1, 1, W]
        cur = torch.cat([cur, next_row], dim=1)
        out.append(next_row.squeeze(0).squeeze(0).to(torch.long).cpu())

    return torch.stack(out, dim=0)


def rollout_rule30_last_window(
    initial: torch.Tensor, total_steps: int, window_steps: int, extra_margin: int = 0
) -> torch.Tensor:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if window_steps < 1:
        raise ValueError("window_steps must be >= 1")

    width = initial.numel()
    growth_margin = max(0, total_steps - 1)
    margin = growth_margin + max(0, extra_margin)
    full_width = width + 2 * margin

    cur_full = torch.zeros(full_width, dtype=torch.long)
    start = margin
    end = start + width
    cur_full[start:end] = initial.clone().long()

    window = deque(maxlen=window_steps)
    window.append(cur_full[start:end].clone())
    for _ in range(total_steps - 1):
        cur_full = rule30_next_row(cur_full)
        window.append(cur_full[start:end].clone())
    return torch.stack(list(window), dim=0)


@torch.no_grad()
def rollout_model_last_window_blocked(
    model: nn.Module,
    initial: torch.Tensor,
    total_steps: int,
    context_steps: int,
    block_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if context_steps < 2:
        raise ValueError("context_steps must be >= 2")
    if block_steps < 1:
        raise ValueError("block_steps must be >= 1")

    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    context_max = context_steps - 1
    width = initial.numel()

    # Fixed-size circular context buffer on GPU:
    # new rows stay in memory, old rows are overwritten after overflow.
    ctx_buf = torch.empty((context_max, width), device=device, dtype=torch.float32)
    ctx_buf[0] = initial.to(device=device, dtype=torch.float32)
    valid_len = 1
    start = 0  # index of oldest row in circular buffer when full

    window = deque(maxlen=context_steps)
    window.append(initial.clone().long())

    produced = 1
    while produced < total_steps:
        to_generate = min(block_steps, total_steps - produced)
        for _ in range(to_generate):
            if valid_len < context_max:
                ctx = ctx_buf[:valid_len].unsqueeze(0)  # [1, T, W]
            else:
                order = (torch.arange(context_max, device=device) + start) % context_max
                ctx = ctx_buf.index_select(0, order).unsqueeze(0)  # [1, T, W]

            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(ctx)
            next_row = (logits[:, -1, :] > 0).to(torch.float32).squeeze(0)

            if valid_len < context_max:
                ctx_buf[valid_len] = next_row
                valid_len += 1
            else:
                ctx_buf[start] = next_row
                start = (start + 1) % context_max

            window.append(next_row.to(torch.long).cpu())
            produced += 1
    return torch.stack(list(window), dim=0)


def evaluate_rollout(truth: torch.Tensor, pred: torch.Tensor, edge_ignore: int) -> None:
    global_acc = (truth == pred).float().mean().item()
    truth_center = _slice_center(truth, edge_ignore)
    pred_center = _slice_center(pred, edge_ignore)
    center_acc = (truth_center == pred_center).float().mean().item()
    base0_global = (truth == 0).float().mean().item()
    base0_center = (truth_center == 0).float().mean().item()
    print(f"rollout bit accuracy (global): {global_acc:.4f}")
    print(f"rollout bit accuracy (center, ignore={edge_ignore}): {center_acc:.4f}")
    print(f"rollout baseline always-0 (global): {base0_global:.4f}")
    print(f"rollout baseline always-0 (center): {base0_center:.4f}")


def grid_to_photoimage(grid: torch.Tensor, cell: int) -> ImageTk.PhotoImage:
    arr = grid.to(torch.uint8).cpu().numpy()
    img_arr = (1 - arr) * 255
    img = Image.fromarray(img_arr, mode="L")
    if cell > 1:
        img = img.resize((img.width * cell, img.height * cell), resample=Image.Resampling.NEAREST)
    return ImageTk.PhotoImage(img)


def show_ui(truth: torch.Tensor, pred: torch.Tensor, cfg: Config) -> None:
    rows, cols = truth.shape
    cell = cfg.cell_size
    pad = 16
    block_w = cols * cell
    w = pad * 3 + block_w * 2
    h = 40 + rows * cell + pad

    root = tk.Tk()
    root.title("Rule 30: exact vs transformer rollout")
    canvas = tk.Canvas(root, width=w, height=h, bg="white", highlightthickness=0)
    canvas.pack()

    truth_img = grid_to_photoimage(truth, cell)
    pred_img = grid_to_photoimage(pred, cell)
    root._img_refs = [truth_img, pred_img]

    canvas.create_text(pad, 12, text="Exact Rule 30", anchor="nw", fill="#1a1a1a")
    canvas.create_text(pad * 2 + block_w, 12, text="Transformer prediction", anchor="nw", fill="#1a1a1a")
    canvas.create_image(pad, 24, image=truth_img, anchor="nw")
    canvas.create_image(pad * 2 + block_w, 24, image=pred_img, anchor="nw")

    root.resizable(False, False)
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.inference_only and not args.weights_in:
        raise ValueError("--inference-only requires --weights-in to load a trained model.")

    cfg = Config()
    cfg.seed = random.SystemRandom().randrange(0, 2**63 - 1)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    print(f"seed: {cfg.seed}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Install CUDA-enabled PyTorch and run on a machine with NVIDIA GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}")
    print(
        f"tf32: enabled | token_cells={cfg.token_cells} | sim_extra_margin={cfg.sim_extra_margin}"
    )

    base_model = TemporalTokenTransformer(cfg).to(device)
    if args.weights_in:
        state = torch.load(args.weights_in, map_location=device)
        base_model.load_state_dict(state)
        print(f"loaded weights from: {args.weights_in}")

    model: nn.Module = base_model
    if args.no_compile:
        print("torch.compile: disabled by --no-compile")
    else:
        try:
            dynamo = importlib.import_module("torch._dynamo")
            dynamo.config.suppress_errors = True
            model = torch.compile(base_model)
            with torch.no_grad():
                dummy = torch.zeros((1, min(4, cfg.steps - 1), cfg.width), device=device, dtype=torch.float32)
                _ = model(dummy)
            print("torch.compile: enabled")
        except Exception as exc:
            model = base_model
            print(f"torch.compile: fallback to eager ({exc})")

    if not args.inference_only:
        print("building dataset...")
        x, y = make_dataset(cfg)
        val_init, val_truth = make_rollout_validation_set(cfg)
        print("training...")
        train_model(
            model,
            x,
            y,
            cfg,
            device,
            val_init=val_init,
            val_truth=val_truth,
            rollout_block_steps=args.inference_block_steps,
        )
        if args.weights_out:
            torch.save(base_model.state_dict(), args.weights_out)
            print(f"saved weights to: {args.weights_out}")
    else:
        print("inference-only mode: training skipped")

    print("evaluating on random init...")
    total_infer_steps = args.inference_total_steps if args.inference_total_steps is not None else cfg.steps
    if total_infer_steps < cfg.steps:
        raise ValueError(
            f"--inference-total-steps must be >= cfg.steps ({cfg.steps}), got {total_infer_steps}"
        )
    init = torch.randint(0, 2, (cfg.width,), dtype=torch.long)

    t0 = time.perf_counter()
    truth = rollout_rule30_last_window(
        init, total_infer_steps, cfg.steps, extra_margin=cfg.sim_extra_margin
    )
    exact_sec = time.perf_counter() - t0

    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    pred = rollout_model_last_window_blocked(
        model,
        init,
        total_steps=total_infer_steps,
        context_steps=cfg.steps,
        block_steps=args.inference_block_steps,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    nn_sec = time.perf_counter() - t1

    speed_ratio = nn_sec / max(exact_sec, 1e-12)
    print(
        f"inference rollout: total_steps={total_infer_steps}, shown_last_window={cfg.steps}, "
        f"nn_block_steps={args.inference_block_steps}"
    )
    print(f"inference time exact_rule30: {exact_sec:.6f} sec")
    print(f"inference time neural_rollout: {nn_sec:.6f} sec")
    print(f"time ratio neural/exact: {speed_ratio:.2f}x")

    evaluate_rollout(truth, pred, cfg.edge_ignore)
    show_ui(truth, pred, cfg)


if __name__ == "__main__":
    main()
