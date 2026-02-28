"""
Python 3.11
Rule 30 + PyTorch Transformer demo:
1) Generate many Rule 30 rollouts from random initial states.
2) Train a small Transformer to predict next row from current row.
3) Roll out model predictions autoregressively and compare with exact simulation.
4) Show both in a simple Tkinter UI.
"""

from __future__ import annotations

import argparse
import random
import tkinter as tk
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def rule30_next_row(row: torch.Tensor) -> torch.Tensor:
    # row shape: [W], values in {0,1}
    left = torch.zeros_like(row)
    left[1:] = row[:-1]
    center = row
    right = torch.zeros_like(row)
    right[:-1] = row[1:]
    pattern = (left << 2) | (center << 1) | right
    return ((30 >> pattern) & 1).to(torch.long)


def rollout_rule30(initial: torch.Tensor, steps: int) -> torch.Tensor:
    # returns [steps, W]
    out = [initial.clone().long()]
    cur = initial.clone().long()
    for _ in range(steps - 1):
        cur = rule30_next_row(cur)
        out.append(cur.clone())
    return torch.stack(out, dim=0)


@dataclass
class Config:
    width: int = 512
    steps: int = 512
    train_sequences: int = 256
    batch_size: int = 16
    epochs: int = 15
    lr: float = 3e-4
    d_model: int = 1024
    nhead: int = 16
    layers: int = 18
    ff_dim: int = 4096
    dropout: float = 0.1
    seed: int = 42
    cell_size: int = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule 30 transformer experiment")
    parser.add_argument("--weights-in", type=str, default=None, help="Path to load model weights (.pt/.pth)")
    parser.add_argument("--weights-out", type=str, default=None, help="Path to save model weights (.pt/.pth)")
    parser.add_argument("--inference-only", action="store_true", help="Run inference/evaluation only (skip training)")
    return parser.parse_args()


class RowTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.bit_embed = nn.Embedding(2, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.width, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.layers)
        self.head = nn.Linear(cfg.d_model, 2)
        self.register_buffer("pos_ids", torch.arange(cfg.width).long(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W] ints 0/1
        h = self.bit_embed(x) + self.pos_embed(self.pos_ids.unsqueeze(0))
        h = self.encoder(h)
        return self.head(h)  # [B, W, 2]


def make_dataset(cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for _ in range(cfg.train_sequences):
        init = torch.randint(0, 2, (cfg.width,), dtype=torch.long)
        traj = rollout_rule30(init, cfg.steps)
        xs.append(traj[:-1])
        ys.append(traj[1:])
    x = torch.cat(xs, dim=0)  # [N*(steps-1), W]
    y = torch.cat(ys, dim=0)  # [N*(steps-1), W]
    return x, y


def train_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: Config, device: torch.device) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    n = x.size(0)
    indices = torch.arange(n)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    print(f"mixed precision: {amp_dtype} (scaler={use_scaler})")
    for epoch in range(1, cfg.epochs + 1):
        indices = indices[torch.randperm(n)]
        total_loss = 0.0
        total_correct = 0
        total_bits = 0
        for i in range(0, n, cfg.batch_size):
            batch_idx = indices[i : i + cfg.batch_size]
            xb = x[batch_idx].to(device, non_blocking=True)
            yb = y[batch_idx].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, 2), yb.view(-1))
            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=-1)
            total_correct += (pred == yb).sum().item()
            total_bits += yb.numel()
        print(
            f"epoch {epoch:02d} | loss={total_loss / n:.4f} | bit_acc={total_correct / total_bits:.4f}",
            flush=True,
        )


@torch.no_grad()
def rollout_model(model: nn.Module, initial: torch.Tensor, steps: int, device: torch.device) -> torch.Tensor:
    model.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    cur = initial.clone().to(device).unsqueeze(0)  # [1, W]
    out = [cur.squeeze(0).cpu()]
    for _ in range(steps - 1):
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits = model(cur)
        cur = logits.argmax(dim=-1)  # [1, W]
        out.append(cur.squeeze(0).cpu())
    return torch.stack(out, dim=0)


def draw_grid(canvas: tk.Canvas, grid: torch.Tensor, x_offset: int, title: str, cell: int) -> None:
    rows, cols = grid.shape
    canvas.create_text(x_offset, 12, text=title, anchor="nw", fill="#1a1a1a")
    top = 24
    for r in range(rows):
        y0 = top + r * cell
        y1 = y0 + cell
        for c in range(cols):
            x0 = x_offset + c * cell
            x1 = x0 + cell
            color = "#111111" if int(grid[r, c]) else "#f2f2f2"
            canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")


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
    draw_grid(canvas, truth, pad, "Exact Rule 30", cell)
    draw_grid(canvas, pred, pad * 2 + block_w, "Transformer prediction", cell)
    root.resizable(False, False)
    root.mainloop()


def main() -> None:
    args = parse_args()
    if args.inference_only and not args.weights_in:
        raise ValueError("--inference-only requires --weights-in to load a trained model.")
    cfg = Config()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Install CUDA-enabled PyTorch and run on a machine with NVIDIA GPU.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"device: {device}")
    model = RowTransformer(cfg).to(device)
    if args.weights_in:
        state = torch.load(args.weights_in, map_location=device)
        model.load_state_dict(state)
        print(f"loaded weights from: {args.weights_in}")
    if not args.inference_only:
        print("building dataset...")
        x, y = make_dataset(cfg)
        print("training...")
        train_model(model, x, y, cfg, device)
        if args.weights_out:
            torch.save(model.state_dict(), args.weights_out)
            print(f"saved weights to: {args.weights_out}")
    else:
        print("inference-only mode: training skipped")

    print("evaluating on single-center init...")
    init = torch.zeros(cfg.width, dtype=torch.long)
    init[cfg.width // 2] = 1
    truth = rollout_rule30(init, cfg.steps)
    pred = rollout_model(model, init, cfg.steps, device)
    bit_acc = (truth == pred).float().mean().item()
    print(f"rollout bit accuracy (single-seed test): {bit_acc:.4f}")
    show_ui(truth, pred, cfg)


if __name__ == "__main__":
    main()
