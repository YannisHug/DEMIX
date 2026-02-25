"""
Script d'entraînement du Mini U-Net de séparation audio.

Usage :
    python train.py --musdb_root /path/to/musdb18 --source vocals --epochs 20

Le script :
1. Charge les données MUSDB18 (subset réduit)
2. Entraîne un SourceUNet sur la source choisie
3. Sauvegarde le meilleur modèle (val loss minimale)
4. Exporte les courbes de loss (loss_curves.png)
"""
import static_ffmpeg
static_ffmpeg.add_paths()

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ── Imports locaux ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.unet import SourceUNet
from data.dataset import get_dataloaders


# ── Hyperparamètres par défaut ────────────────────────────────────────────────
DEFAULTS = dict(
    musdb_root="data/musdb18-small",
    source="vocals",
    train_tracks=25,
    val_tracks=5,
    chunks_per_track=10,
    batch_size=8,
    epochs=20,
    lr=1e-3,
    loss="l1",          # 'l1' ou 'mse'
    num_workers=2,
    save_dir="./outputs",
    device="auto",
)


def get_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """L1 ou MSE entre spectrogramme prédit et spectrogramme cible."""
    # Trim pour aligner les dimensions (légères différences possibles)
    t = min(pred.shape[-1], target.shape[-1])
    f = min(pred.shape[-2], target.shape[-2])
    return criterion(pred[:, :, :f, :t], target[:, :, :f, :t])


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for i, (mix_spec, tgt_spec) in enumerate(loader):
        mix_spec = mix_spec.to(device)
        tgt_spec = tgt_spec.to(device)

        optimizer.zero_grad()
        pred = model(mix_spec)
        loss = compute_loss(pred, tgt_spec, criterion)
        loss.backward()

        # Gradient clipping (stabilité)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % max(1, n_batches // 5) == 0:
            print(f"  Epoch {epoch} [{i+1}/{n_batches}] "
                  f"batch_loss={loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for mix_spec, tgt_spec in loader:
        mix_spec = mix_spec.to(device)
        tgt_spec = tgt_spec.to(device)
        pred = model(mix_spec)
        loss = compute_loss(pred, tgt_spec, criterion)
        total_loss += loss.item()
    return total_loss / len(loader)


def plot_curves(train_losses, val_losses, save_path: str):
    """Sauvegarde les courbes de loss train/val."""
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-o", label="Train loss", linewidth=2)
    ax.plot(epochs, val_losses, "r-o", label="Val loss", linewidth=2)
    ax.set_xlabel("Époque")
    ax.set_ylabel(f"Loss")
    ax.set_title("Courbes d'apprentissage — Mini U-Net Demixing")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Courbes sauvegardées : {save_path}")


def main(cfg: dict):
    device = get_device(cfg["device"])
    print(f"\n{'='*60}")
    print(f"  Mini U-Net Source Separation — {cfg['source'].upper()}")
    print(f"  Device : {device}")
    print(f"  Loss   : {cfg['loss'].upper()}")
    print(f"  Epochs : {cfg['epochs']}")
    print(f"{'='*60}\n")

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        musdb_root=cfg["musdb_root"],
        source=cfg["source"],
        train_tracks=cfg["train_tracks"],
        val_tracks=cfg["val_tracks"],
        chunks_per_track=cfg["chunks_per_track"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = SourceUNet(in_channels=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Modèle] SourceUNet — {n_params:,} paramètres\n")

    # ── Optimiseur & Loss ─────────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.L1Loss() if cfg["loss"] == "l1" else nn.MSELoss()

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"\nÉpoque {epoch:03d}/{cfg['epochs']:03d} | "
              f"train={train_loss:.4f} | val={val_loss:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"{elapsed:.1f}s")

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                },
                save_dir / f"best_model_{cfg['source']}.pt",
            )
            print(f"  [✓] Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")

    print(f"\n{'='*60}")
    print(f"  Entraînement terminé !")
    print(f"  Meilleur modèle : époque {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"{'='*60}")

    # ── Export des courbes ────────────────────────────────────────────────────
    plot_curves(
        train_losses, val_losses,
        str(save_dir / f"loss_curves_{cfg['source']}.png"),
    )

    # Sauvegarde des métriques JSON
    metrics = {
        "source": cfg["source"],
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }
    with open(save_dir / f"metrics_{cfg['source']}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Mini U-Net Demixing")
    for key, val in DEFAULTS.items():
        t = type(val) if val is not None else str
        parser.add_argument(f"--{key}", type=t, default=val)

    args = parser.parse_args()
    cfg = vars(args)
    main(cfg)
