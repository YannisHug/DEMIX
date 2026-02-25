"""
Mini U-Net pour la séparation de sources audio.
Architecture inspirée de Spleeter / Jansson et al. (ISMIR 2017),
réimplémentée en PyTorch avec une version allégée (4 couches encoder).

Entrée  : spectrogramme de magnitude STFT du mix  [B, 1, F, T]
Sortie  : masque sigmoid par source               [B, 1, F, T]
Signal  : masque × spectrogramme mix              [B, 1, F, T]
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloc encodeur : Conv2d → BatchNorm → LeakyReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    """Bloc décodeur : ConvTranspose2d → BatchNorm → ReLU → (Dropout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SourceUNet(nn.Module):
    """
    U-Net léger pour UNE source (vocals, drums, bass ou other).

    Encoder : 4 couches Conv2D avec downsampling ×2 à chaque étape.
    Decoder : 4 couches ConvTranspose2D avec skip connections (concat).
    Sortie  : masque sigmoid appliqué au spectrogramme du mix.

    Filtres : [16, 32, 64, 128]  (vs [16,32,64,128,256,512] chez Spleeter)
    → ~10× moins de paramètres, entraînable sur CPU en quelques heures.
    """

    FILTERS = [16, 32, 64, 128]

    def __init__(self, in_channels: int = 2):
        """
        Args:
            in_channels: nombre de canaux d'entrée (1=mono, 2=stéréo).
        """
        super().__init__()
        f = self.FILTERS
        # ── Encodeur ───────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, f[0])   # /2
        self.enc2 = ConvBlock(f[0], f[1])           # /4
        self.enc3 = ConvBlock(f[1], f[2])           # /8
        self.enc4 = ConvBlock(f[2], f[3])           # /16  (bottleneck)

        # ── Décodeur (avec skip connections) ───────────────────────────────
        # Après concat le nombre de canaux double → on divise par 2 en sortie
        self.dec4 = DeconvBlock(f[3], f[2], dropout=0.5)            # /8
        self.dec3 = DeconvBlock(f[2] * 2, f[1], dropout=0.5)        # /4
        self.dec2 = DeconvBlock(f[1] * 2, f[0])                     # /2
        self.dec1 = DeconvBlock(f[0] * 2, in_channels)              # /1

        # ── Couche finale : masque sigmoid ──────────────────────────────────
        self.mask_conv = nn.Conv2d(
            in_channels * 2, in_channels,
            kernel_size=4, padding=2, dilation=2,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, mix_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mix_spec: spectrogramme magnitude du mix [B, C, F, T]
        Returns:
            source_spec: spectrogramme estimé de la source [B, C, F, T]
        """
        # Encodeur
        e1 = self.enc1(mix_spec)   # [B, 16, F/2,  T/2 ]
        e2 = self.enc2(e1)          # [B, 32, F/4,  T/4 ]
        e3 = self.enc3(e2)          # [B, 64, F/8,  T/8 ]
        e4 = self.enc4(e3)          # [B,128, F/16, T/16]  bottleneck

        # Décodeur + skip connections (Concatenate sur l'axe canaux)
        d4 = self.dec4(e4)                                    # [B, 64, F/8,  T/8 ]
        d4 = self._match_and_cat(e3, d4)                     # [B,128, F/8,  T/8 ]

        d3 = self.dec3(d4)                                    # [B, 32, F/4,  T/4 ]
        d3 = self._match_and_cat(e2, d3)                     # [B, 64, F/4,  T/4 ]

        d2 = self.dec2(d3)                                    # [B, 16, F/2,  T/2 ]
        d2 = self._match_and_cat(e1, d2)                     # [B, 32, F/2,  T/2 ]

        d1 = self.dec1(d2)                                    # [B,  C, F,    T   ]
        d1 = self._match_and_cat(mix_spec, d1)               # [B, 2C, F,    T   ]

        # Masque sigmoid × mix
        mask = self.sigmoid(self.mask_conv(d1))               # [B,  C, F',   T'  ]

        # Trim strict sur F et T pour matcher exactement le mix d'entrée
        F_in = mix_spec.shape[2]
        T_in = mix_spec.shape[3]
        mask = mask[:, :, :F_in, :T_in]

        # Pad si le masque est plus petit que le mix (cas rare selon chunk size)
        pad_f = F_in - mask.shape[2]
        pad_t = T_in - mask.shape[3]
        if pad_f > 0 or pad_t > 0:
            mask = torch.nn.functional.pad(mask, (0, pad_t, 0, pad_f))

        return mask * mix_spec

    @staticmethod
    def _match_and_cat(skip: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Crop/pad x pour matcher les dims de skip, puis concatene."""
        # Crop si x est trop grand
        x = x[:, :, :skip.shape[2], :skip.shape[3]]
        return torch.cat([skip, x], dim=1)


class MultiSourceDemixer(nn.Module):
    """
    Modèle complet : un SourceUNet indépendant par source.
    Inspiré de l'approche Spleeter (un réseau par instrument).

    Sources : vocals, drums, bass, other.
    """

    SOURCES = ["vocals", "drums", "bass", "other"]

    def __init__(self, in_channels: int = 2):
        super().__init__()
        self.nets = nn.ModuleDict({
            src: SourceUNet(in_channels=in_channels)
            for src in self.SOURCES
        })

    def forward(self, mix_spec: torch.Tensor) -> dict:
        """
        Args:
            mix_spec: [B, C, F, T]
        Returns:
            dict source → tensor [B, C, F, T]
        """
        return {src: net(mix_spec) for src, net in self.nets.items()}

    def forward_source(self, mix_spec: torch.Tensor, source: str) -> torch.Tensor:
        """Inférence pour une seule source (plus rapide)."""
        return self.nets[source](mix_spec)


if __name__ == "__main__":
    # Test rapide de l'architecture
    B, C, F, T = 2, 2, 512, 128   # batch=2, stéréo, 512 bins freq, 128 trames
    x = torch.randn(B, C, F, T)

    print("=== Test SourceUNet (vocals) ===")
    net = SourceUNet(in_channels=C)
    out = net(x)
    print(f"  Entrée : {x.shape}")
    print(f"  Sortie : {out.shape}")
    n_params = sum(p.numel() for p in net.parameters())
    print(f"  Paramètres : {n_params:,}")

    print("\n=== Test MultiSourceDemixer (4 sources) ===")
    model = MultiSourceDemixer(in_channels=C)
    outputs = model(x)
    for src, t in outputs.items():
        print(f"  {src}: {t.shape}")
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Paramètres totaux : {n_total:,}")