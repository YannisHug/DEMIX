"""
Dataset MUSDB18 pour l'entraînement du modèle de séparation.

Pipeline :
1. Charge une piste (mix + stems) via musdb
2. Calcule le STFT → spectrogramme de magnitude
3. Découpe en segments de durée fixe (chunks)
4. Retourne (mix_spec, target_spec) pour une source donnée
"""

import random
from pathlib import Path
from typing import List, Optional, Tuple

import musdb
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Paramètres STFT ────────────────────────────────────────────────────────────
N_FFT = 4096       # Fenêtre FFT (comme Spleeter / UMX)
HOP_LENGTH = 1024  # Décalage entre trames
SR = 44100         # Fréquence d'échantillonnage MUSDB18
CHUNK_DURATION = 6.0  # Durée d'un segment en secondes
CHUNK_SAMPLES = int(CHUNK_DURATION * SR)


def compute_stft(
    audio: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Calcule le STFT d'un signal stéréo.

    Args:
        audio: [2, N] tableau numpy stéréo
        n_fft: taille fenêtre FFT
        hop_length: décalage entre trames

    Returns:
        magnitude: [2, F, T] avec F = n_fft//2 + 1
    """
    import librosa

    specs = []
    for ch in range(audio.shape[0]):
        stft = librosa.stft(audio[ch], n_fft=n_fft, hop_length=hop_length)
        specs.append(np.abs(stft))  # magnitude uniquement
    return np.stack(specs, axis=0)  # [2, F, T]


class MUSDB18Dataset(Dataset):
    """
    Dataset PyTorch pour MUSDB18.

    Chaque item est un tuple (mix_spec, target_spec) :
    - mix_spec    : spectrogramme magnitude du mix    [2, F, T]
    - target_spec : spectrogramme magnitude de la source cible [2, F, T]

    On découpe chaque piste en segments de CHUNK_DURATION secondes.
    """

    SOURCES = ["vocals", "drums", "bass", "other"]

    def __init__(
        self,
        root: str,
        subset: str = "train",
        source: str = "vocals",
        max_tracks: Optional[int] = None,
        chunks_per_track: int = 10,
        seed: int = 42,
    ):
        """
        Args:
            root: chemin vers le dossier MUSDB18
            subset: 'train' ou 'test'
            source: source cible parmi SOURCES
            max_tracks: nombre max de pistes à charger (None = toutes)
            chunks_per_track: nombre de segments aléatoires par piste
            seed: graine pour la reproductibilité
        """
        assert source in self.SOURCES, f"Source inconnue : {source}"

        self.root = root
        self.source = source
        self.chunks_per_track = chunks_per_track
        self.rng = random.Random(seed)

        # Chargement du dataset musdb
        db = musdb.DB(root=root, subsets=subset, is_wav=False)
        tracks = list(db.tracks)

        if max_tracks is not None:
            tracks = tracks[:max_tracks]

        print(f"[Dataset] {subset} | source={source} | {len(tracks)} pistes | "
              f"{chunks_per_track} chunks/piste = {len(tracks)*chunks_per_track} items")

        # Pré-calcul des spectrogrammes (mis en cache en mémoire)
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self._build(tracks)

    def _build(self, tracks) -> None:
        """Charge et découpe toutes les pistes."""
        import librosa

        for track in tracks:
            mix_audio = track.audio.T.astype(np.float32)         # [2, N]
            target_audio = track.targets[self.source].audio.T.astype(np.float32)

            n_samples = mix_audio.shape[1]
            if n_samples < CHUNK_SAMPLES:
                continue

            # Générer chunks_per_track segments aléatoires
            for _ in range(self.chunks_per_track):
                start = self.rng.randint(0, n_samples - CHUNK_SAMPLES)
                end = start + CHUNK_SAMPLES

                mix_chunk = mix_audio[:, start:end]
                tgt_chunk = target_audio[:, start:end]

                mix_spec = compute_stft(mix_chunk)   # [2, F, T]
                tgt_spec = compute_stft(tgt_chunk)   # [2, F, T]

                self.samples.append((mix_spec, tgt_spec))

        print(f"[Dataset] {len(self.samples)} segments chargés en mémoire")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mix_spec, tgt_spec = self.samples[idx]
        return (
            torch.from_numpy(mix_spec).float(),
            torch.from_numpy(tgt_spec).float(),
        )


def get_dataloaders(
    musdb_root: str,
    source: str = "vocals",
    train_tracks: int = 25,
    val_tracks: int = 5,
    chunks_per_track: int = 10,
    batch_size: int = 8,
    num_workers: int = 2,
):
    """
    Construit les DataLoaders train et validation.

    Args:
        musdb_root: chemin MUSDB18
        source: source cible
        train_tracks: nombre de pistes d'entraînement
        val_tracks: nombre de pistes de validation
        chunks_per_track: segments par piste
        batch_size: taille du batch
        num_workers: workers DataLoader

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader

    train_set = MUSDB18Dataset(
        root=musdb_root,
        subset="train",
        source=source,
        max_tracks=train_tracks,
        chunks_per_track=chunks_per_track,
    )
    val_set = MUSDB18Dataset(
        root=musdb_root,
        subset="test",  # on split le train en train/val
        source=source,
        max_tracks=train_tracks + val_tracks,
        chunks_per_track=5,  # moins de chunks pour la validation
    )
    # On garde uniquement les dernières pistes pour la validation
    val_set.samples = val_set.samples[-(val_tracks * 5):]

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
