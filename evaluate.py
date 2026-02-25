"""
Évaluation BSS-Eval par segments de 30s — compatible NumPy 2.x, sans museval.
Calcule SDR, SI-SDR et SAR. SIR ignoré.
"""

import static_ffmpeg
static_ffmpeg.add_paths()

import argparse
import json
from pathlib import Path

import musdb
import mir_eval
import numpy as np
import torch
import librosa
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.unet import SourceUNet
from data.dataset import compute_stft, N_FFT, HOP_LENGTH, SR


@torch.no_grad()
def separate(model, mix_audio, device, chunk_sec=6):
    """Séparation chunk par chunk avec reconstruction ISTFT."""
    chunk_samples = SR * chunk_sec
    n = mix_audio.shape[1]
    estimated = np.zeros_like(mix_audio)

    for pos in range(0, n, chunk_samples):
        end = min(pos + chunk_samples, n)
        chunk = mix_audio[:, pos:end]
        pad = chunk_samples - chunk.shape[1]
        if pad > 0:
            chunk = np.pad(chunk, ((0, 0), (0, pad)))

        spec = compute_stft(chunk, N_FFT, HOP_LENGTH)
        tensor = torch.from_numpy(spec).float().unsqueeze(0).to(device)
        pred_spec = model(tensor).cpu().numpy()[0]

        for ch in range(2):
            mix_stft = librosa.stft(chunk[ch], n_fft=N_FFT, hop_length=HOP_LENGTH)
            phase = np.exp(1j * np.angle(mix_stft))
            t = min(pred_spec.shape[-1], mix_stft.shape[-1])
            f = min(pred_spec.shape[-2], mix_stft.shape[-2])
            pred_stft = pred_spec[ch, :f, :t] * phase[:f, :t]
            audio_ch = librosa.istft(pred_stft, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                     length=chunk_samples)
            estimated[ch, pos:end] += audio_ch[:end - pos]

    return estimated


def si_sdr(ref, est):
    """Calcule le Scale-Invariant SDR pour un signal mono."""
    ref = ref.astype(np.float32)
    est = est.astype(np.float32)
    alpha = np.sum(ref * est) / np.sum(ref**2)
    e_true = alpha * ref
    e_res = est - e_true
    return 10 * np.log10(np.sum(e_true**2) / np.sum(e_res**2))


def bss_eval_chunked(ref, est, chunk_sec=30):
    """
    Calcule SDR, SI-SDR et SAR sur des chunks de chunk_sec secondes puis moyenne.
    Travaille sur le canal gauche uniquement (mono).
    ref, est : [2, N]
    """
    chunk_samples = SR * chunk_sec
    n = min(ref.shape[1], est.shape[1])

    ref_mono = ref[0, :n]
    est_mono = est[0, :n]

    sdrs, si_sdrs, sars = [], [], []

    for pos in range(0, n, chunk_samples):
        end = min(pos + chunk_samples, n)
        if end - pos < SR * 1:
            continue

        r = ref_mono[pos:end]
        e = est_mono[pos:end]

        if np.max(np.abs(r)) < 1e-4:
            continue

        # SDR classique + SAR
        try:
            sdr, _, sar, _ = mir_eval.separation.bss_eval_sources(
                r[np.newaxis, :],
                e[np.newaxis, :],
            )
            if np.isfinite(sdr[0]) and np.isfinite(sar[0]):
                sdrs.append(float(sdr[0]))
                sars.append(float(sar[0]))
        except Exception:
            continue

        # SI-SDR
        try:
            si_sdr_val = si_sdr(r, e)
            if np.isfinite(si_sdr_val):
                si_sdrs.append(float(si_sdr_val))
        except Exception:
            continue

    if not sdrs:
        return None, None, None

    sdr_mean = float(np.mean(sdrs))
    si_sdr_mean = float(np.mean(si_sdrs)) if si_sdrs else float('nan')
    sar_mean = float(np.mean(sars))
    return sdr_mean, si_sdr_mean, sar_mean


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Évaluation] source={args.source} | device={device} | {args.n_tracks} pistes")
    print(f"[Info] Calcul BSS-Eval par segments de 30s (canal gauche)\n")

    # Chargement modèle
    ckpt = torch.load(args.model_path, map_location=device)
    model = SourceUNet(in_channels=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[✓] Modèle chargé (époque {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})\n")

    # MUSDB18
    try:
        db = musdb.DB(root=args.musdb_root, subsets="test", is_wav=False)
        tracks = list(db.tracks)[:args.n_tracks]
    except Exception:
        tracks = []
    if not tracks:
        print("[!] Pas de pistes test, fallback sur train")
        db = musdb.DB(root=args.musdb_root, subsets="train", is_wav=False)
        tracks = list(db.tracks)[:args.n_tracks]

    baseline_scores, model_scores = [], []

    for i, track in enumerate(tracks):
        print(f"Track {i+1}/{len(tracks)}: {track.name}")
        mix = track.audio.T.astype(np.float32)
        ref = track.targets[args.source].audio.T.astype(np.float32)

        if np.max(np.abs(ref)) < 1e-4:
            print(f"  [!] Source silencieuse, ignorée\n")
            continue

        # Baseline
        sdr_b, si_sdr_b, sar_b = bss_eval_chunked(ref, mix)
        if sdr_b is None:
            print("  [!] Calcul baseline échoué\n")
            continue
        baseline_scores.append((sdr_b, si_sdr_b, sar_b))
        print(f"  Baseline  → SDR={sdr_b:6.2f} | SI-SDR={si_sdr_b:6.2f} | SAR={sar_b:6.2f}")

        # Notre modèle
        print(f"  Séparation en cours...", end=" ", flush=True)
        est = separate(model, mix, device)
        sdr_m, si_sdr_m, sar_m = bss_eval_chunked(ref, est)
        if sdr_m is None:
            print("calcul échoué\n")
            continue
        model_scores.append((sdr_m, si_sdr_m, sar_m))
        print(f"OK")
        print(f"  Our model → SDR={sdr_m:6.2f} | SI-SDR={si_sdr_m:6.2f} | SAR={sar_m:6.2f}\n")

    if not model_scores:
        print("[!] Aucune piste évaluable.")
        return

    def avg(scores, idx):
        vals = [s[idx] for s in scores if s[idx] is not None and np.isfinite(s[idx])]
        return float(np.mean(vals)) if vals else float('nan')

    print("=" * 58)
    print(f"  RÉSUMÉ — {args.source.upper()} ({len(model_scores)} pistes)")
    print("=" * 58)
    print(f"  {'Système':<20} {'SDR':>7} {'SI-SDR':>7} {'SAR':>7}")
    print(f"  {'-'*45}")
    print(f"  {'Baseline naïf':<20} {avg(baseline_scores,0):>7.2f} {avg(baseline_scores,1):>7.2f} {avg(baseline_scores,2):>7.2f}")
    print(f"  {'Notre modèle':<20} {avg(model_scores,0):>7.2f} {avg(model_scores,1):>7.2f} {avg(model_scores,2):>7.2f}")

    umx = {"vocals":(6.32,13.33,6.52),"drums":(5.73,11.12,6.02),
           "bass":(5.23,10.93,6.34),"other":(4.02,6.59,5.73)}
    if args.source in umx:
        r = umx[args.source]
        print(f"  {'UMX (référence)':<20} {r[0]:>7.2f} {r[1]:>7.2f} {r[2]:>7.2f}")
    print("=" * 58)

    summary = {
        "source": args.source,
        "n_tracks": len(model_scores),
        "baseline": {"SDR": avg(baseline_scores,0), "SI-SDR": avg(baseline_scores,1), "SAR": avg(baseline_scores,2)},
        "our_model": {"SDR": avg(model_scores,0), "SI-SDR": avg(model_scores,1), "SAR": avg(model_scores,2)},
        "umx_reference": dict(zip(["SDR","SI-SDR","SAR"], umx.get(args.source,(None,None,None)))),
    }
    out = Path(args.model_path).parent / f"eval_{args.source}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Résultats sauvegardés : {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--source", default="vocals")
    parser.add_argument("--n_tracks", type=int, default=5)
    main(parser.parse_args())