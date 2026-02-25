"""
Application Streamlit — Démo de séparation audio avec le Mini U-Net.

Fonctionnalités :
- Upload d'un fichier MP4 MUSDB18 (multicanal)
- Extraction automatique du mix et des 4 stems de référence
- Séparation en 4 stems (vocals, drums, bass, other)
- Ecoute de chaque stem dans l'interface
- Visualisation des spectrogrammes
- Comparaison avec Open-Unmix (UMX)
- Métrique SI-SDR calculée automatiquement depuis les stems embarqués

Format MP4 MUSDB18 (10 canaux) :
  ch 0-1  : mix
  ch 2-3  : drums
  ch 4-5  : bass
  ch 6-7  : other
  ch 8-9  : vocals

Lancement :
    streamlit run app.py
"""

import io
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Audio Demixing — Mini U-Net",
    page_icon=None,
    layout="wide",
)

# ── CSS : image de fond sombre + styles généraux ──────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Unbounded:wght@400;700;900&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    /* Fond global avec image assombrie */
    .stApp {
        background-image:
            linear-gradient(rgba(5, 5, 10, 0.82), rgba(5, 5, 10, 0.88)),
            url("https://images.unsplash.com/photo-1598488035139-bdbb2231ce04?w=1800&q=80");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        font-family: 'IBM Plex Sans', sans-serif;
        color: #e8e4dc;
    }

    /* Typographie principale */
    h1, h2, h3, h4 {
        font-family: 'Unbounded', sans-serif !important;
        letter-spacing: -0.5px;
    }

    h1 {
        font-size: 2rem !important;
        font-weight: 900 !important;
        color: #f0ebe0 !important;
        border-bottom: 2px solid #c8a96e;
        padding-bottom: 0.4rem;
        margin-bottom: 0.2rem !important;
        line-height: 1.2 !important;
    }

    h3 {
        color: #c8a96e !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 2rem !important;
    }

    h4 {
        color: #e8e4dc !important;
        font-size: 0.85rem !important;
        font-weight: 700 !important;
        margin-top: 1.2rem !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(8, 8, 14, 0.92) !important;
        border-right: 1px solid rgba(200, 169, 110, 0.2);
    }

    [data-testid="stSidebar"] * {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: #c8c4bc !important;
    }

    [data-testid="stSidebar"] h1 {
        font-family: 'Unbounded', sans-serif !important;
        color: #c8a96e !important;
        border-bottom: 1px solid rgba(200, 169, 110, 0.3) !important;
        font-size: 0.9rem !important;
    }

    /* Blocs info / warning / success */
    .stAlert {
        background: rgba(10, 10, 18, 0.7) !important;
        border-left: 3px solid #c8a96e !important;
        border-radius: 0 !important;
        font-family: 'DM Mono', monospace;
    }

    /* Métriques */
    [data-testid="metric-container"] {
        background: rgba(10, 10, 18, 0.6);
        border: 1px solid rgba(200, 169, 110, 0.25);
        padding: 12px 16px;
        border-radius: 2px;
    }

    [data-testid="stMetricLabel"] {
        color: #c8a96e !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    [data-testid="stMetricValue"] {
        color: #f0ebe0 !important;
        font-family: 'Unbounded', sans-serif !important;
        font-size: 1.3rem !important;
    }

    /* Bouton principal */
    .stButton > button[kind="primary"] {
        background: #c8a96e !important;
        color: #050508 !important;
        border: none !important;
        border-radius: 1px !important;
        font-family: 'Unbounded', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.7rem !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #e0c080 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(200, 169, 110, 0.3) !important;
    }

    /* Boutons secondaires */
    .stButton > button {
        background: rgba(10, 10, 18, 0.6) !important;
        color: #e8e4dc !important;
        border: 1px solid rgba(200, 169, 110, 0.3) !important;
        border-radius: 1px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.8rem !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(10, 10, 18, 0.5) !important;
        border: 1px dashed rgba(200, 169, 110, 0.4) !important;
        border-radius: 2px !important;
    }

    /* Selectbox / text_input */
    [data-testid="stSelectbox"] > div, [data-testid="stTextInput"] > div {
        background: rgba(10, 10, 18, 0.7) !important;
        border: 1px solid rgba(200, 169, 110, 0.25) !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: rgba(10, 10, 18, 0.55) !important;
        border: 1px solid rgba(200, 169, 110, 0.2) !important;
        border-radius: 2px !important;
    }

    /* Progress bar */
    [data-testid="stProgressBar"] > div > div {
        background: #c8a96e !important;
    }

    /* Séparateur */
    hr {
        border-color: rgba(200, 169, 110, 0.2) !important;
        margin: 1.5rem 0 !important;
    }

    /* Texte markdown générique */
    p, li, td, th {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: #c8c4bc !important;
        font-size: 0.88rem !important;
    }

    /* Tableau markdown */
    table {
        border-collapse: collapse;
        width: 100%;
    }
    th {
        border-bottom: 1px solid rgba(200, 169, 110, 0.4) !important;
        color: #c8a96e !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    td, th {
        padding: 6px 12px !important;
    }

    /* Sous-titre de l'app */
    .app-subtitle {
        color: #888 !important;
        font-size: 0.82rem !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 300 !important;
        margin-top: -0.5rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.5px;
    }

    .model-badge {
        display: inline-block;
        background: rgba(200, 169, 110, 0.15);
        border: 1px solid rgba(200, 169, 110, 0.4);
        color: #c8a96e;
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.65rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 1px;
        margin-bottom: 0.5rem;
    }

    .result-card {
        background: rgba(5, 5, 10, 0.7);
        border: 1px solid rgba(200, 169, 110, 0.2);
        border-radius: 2px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .compare-label {
        font-family: 'Unbounded', sans-serif;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #888;
        margin-bottom: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Imports locaux ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))
from models.unet import SourceUNet, MultiSourceDemixer
from data.dataset import compute_stft, N_FFT, HOP_LENGTH, SR


# ── Métriques ─────────────────────────────────────────────────────────────────

def si_sdr(reference: np.ndarray, estimated: np.ndarray, eps: float = 1e-8) -> float:
    """
    Scale-Invariant Signal-to-Distortion Ratio (SI-SDR), en dB.
    Travaille sur signal mono 1-D. Si les deux entrées sont stéréo [2, N],
    retourne la moyenne sur les deux canaux.
    """
    def _si_sdr_1d(ref, est):
        ref = ref - ref.mean()
        est = est - est.mean()
        alpha = (est @ ref) / (ref @ ref + eps)
        proj = alpha * ref
        noise = est - proj
        return 10 * np.log10((proj @ proj + eps) / (noise @ noise + eps))

    if reference.ndim == 2:
        return float(np.mean([
            _si_sdr_1d(reference[ch], estimated[ch]) for ch in range(reference.shape[0])
        ]))
    return _si_sdr_1d(reference, estimated)


# ── Chargement des modèles ────────────────────────────────────────────────────

@st.cache_resource
def load_unet_model(model_path: str):
    """Charge un SourceUNet depuis un checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location=device)
    model = SourceUNet(in_channels=2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device), device


@st.cache_resource
def load_openunmix_model():
    """
    Charge Open-Unmix UMX-L avec les 4 sources simultanement.
    L'algorithme EM interne requiert au minimum 2 targets — on charge
    donc toujours les 4 pour rester dans les conditions nominales.
    """
    try:
        import openunmix
        separator = openunmix.umxl(
            targets=["vocals", "drums", "bass", "other"],
            device="cpu",
        )
        separator.eval()
        return separator
    except ImportError:
        return None


# ── Traitement audio ──────────────────────────────────────────────────────────

# Mapping MUSDB18 : nom du stem -> index du stream audio dans le MP4
# Le conteneur MP4 MUSDB18 contient 5 streams audio stereo independants :
#   stream 0 : mix
#   stream 1 : drums
#   stream 2 : bass
#   stream 3 : other
#   stream 4 : vocals
STEM_STREAMS = {
    "mix":    0,
    "drums":  1,
    "bass":   2,
    "other":  3,
    "vocals": 4,
}


def _ffmpeg_extract_stream(mp4_path: str, stream_index: int, sr: int) -> np.ndarray:
    """
    Extrait un stream audio stereo depuis un fichier MP4 via ffmpeg.
    Retourne un tableau [2, N] float32.
    """
    wav_tmp = mp4_path + f"_stream{stream_index}.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", mp4_path,
                "-map", f"0:a:{stream_index}",   # selection du stream audio N
                "-ac", "2",                        # forcer stereo
                "-ar", str(sr),
                "-f", "wav",
                wav_tmp,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        audio, file_sr = sf.read(wav_tmp, always_2d=True)  # [N, 2]
        audio = audio.T  # [2, N]
        if file_sr != sr:
            audio = np.stack(
                [librosa.resample(audio[ch], orig_sr=file_sr, target_sr=sr) for ch in range(2)],
                axis=0,
            )
        return audio.astype(np.float32)
    finally:
        if os.path.exists(wav_tmp):
            os.unlink(wav_tmp)


def _probe_audio_streams(mp4_path: str) -> int:
    """Retourne le nombre de streams audio dans le fichier via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "a",
                mp4_path,
            ],
            capture_output=True, text=True, check=True,
        )
        import json
        data = json.loads(result.stdout)
        return len(data.get("streams", []))
    except Exception:
        return 0


def extract_stems_from_mp4(uploaded_file, max_sec: int = 30) -> dict:
    """
    Lit un fichier MP4 MUSDB18 et retourne un dict {stem_name: np.ndarray [2, N]}.

    Structure MUSDB18 : 5 streams audio stereo independants dans le conteneur MP4.
    Chaque stream est extrait separement via ffmpeg (-map 0:a:N).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        n_streams = _probe_audio_streams(tmp_path)

        if n_streams == 0:
            st.error(
                "Aucun stream audio detecte. "
                "Verifiez que ffprobe est installe (`apt install ffmpeg`)."
            )
            st.stop()

        if n_streams < 5:
            st.warning(
                f"Le fichier contient {n_streams} stream(s) audio au lieu de 5. "
                "Il ne s'agit peut-etre pas d'un fichier MUSDB18 complet. "
                "Seuls les stems disponibles seront extraits."
            )

        max_samples = SR * max_sec
        stems = {}

        for name, idx in STEM_STREAMS.items():
            if idx >= n_streams:
                continue
            audio = _ffmpeg_extract_stream(tmp_path, idx, SR)
            # Troncature
            if audio.shape[1] > max_samples:
                audio = audio[:, :max_samples]
            stems[name] = audio

        if max_samples < SR * 9999 and any(
            s.shape[1] >= max_samples for s in stems.values()
        ):
            st.info(f"Audio tronque a {max_sec} secondes pour la demo.")

    finally:
        os.unlink(tmp_path)

    return stems


@torch.no_grad()
def separate_unet(
    model: SourceUNet,
    mix_audio: np.ndarray,
    device: torch.device,
    chunk_sec: int = 6,
) -> np.ndarray:
    """Séparation chunk par chunk avec le Mini U-Net."""
    chunk_samples = SR * chunk_sec
    n = mix_audio.shape[1]
    estimated = np.zeros_like(mix_audio)
    n_chunks = int(np.ceil(n / chunk_samples))
    progress = st.progress(0)

    for i, pos in enumerate(range(0, n, chunk_samples)):
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
            actual_len = end - pos
            estimated[ch, pos:end] += audio_ch[:actual_len]

        progress.progress((i + 1) / n_chunks)

    progress.empty()
    return estimated


# Ordre fixe des sources dans la sortie tenseur d'Open-Unmix
UMX_TARGETS = ["vocals", "drums", "bass", "other"]


@torch.no_grad()
def separate_openunmix(separator, mix_audio: np.ndarray, source: str) -> np.ndarray:
    """
    Separation avec Open-Unmix (4 sources chargees simultanement).
    L'EM interne de UMX necessite >= 2 targets ; on separe toujours les 4
    et on retourne uniquement la source demandee.

    UMX retourne un tenseur [n_targets, batch, channels, samples],
    on indexe par la position de la source dans UMX_TARGETS.

    mix_audio : [2, N] float32
    source    : 'vocals' | 'drums' | 'bass' | 'other'
    Retourne  : [2, N] float32
    """
    tensor = torch.from_numpy(mix_audio).float().unsqueeze(0)  # [1, 2, N]
    estimates = separator(tensor)  # tenseur [n_targets, 1, 2, N]
    idx = UMX_TARGETS.index(source)
    return estimates[idx, 0].cpu().numpy()  # [2, N]


def numpy_to_audio_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    """Convertit [2, N] en bytes WAV."""
    buf = io.BytesIO()
    sf.write(buf, audio.T, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def plot_spectrogram(audio: np.ndarray, title: str, sr: int = SR) -> plt.Figure:
    """Spectrogramme (canal gauche) sur fond transparent."""
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0.3))

    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio[0], n_fft=2048, hop_length=512)), ref=np.max
    )
    librosa.display.specshow(D, sr=sr, hop_length=512, x_axis="time",
                             y_axis="log", ax=ax, cmap="inferno")
    ax.set_title(title, color="#c8a96e", fontsize=10, pad=6)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    plt.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB").ax.yaxis.set_tick_params(color="#888")
    plt.tight_layout()
    return fig


# ── Interface principale ──────────────────────────────────────────────────────

def main():
    st.title("Audio Source Separation")
    st.markdown(
        '<p class="app-subtitle">Mini U-Net convolutionnel — entraîné sur MUSDB18 '
        '— comparaison Open-Unmix (UMX-L)</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("Configuration")

    model_dir = st.sidebar.text_input("Dossier des modeles", value="./outputs")
    source_choice = st.sidebar.selectbox(
        "Source a separer",
        options=["vocals", "drums", "bass", "other", "all (4 sources)"],
    )
    compare_umx = st.sidebar.checkbox("Comparer avec Open-Unmix (UMX-L)", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Architecture Mini U-Net**")
    st.sidebar.markdown("- 4 couches Conv2D encoder")
    st.sidebar.markdown("- Skip connections")
    st.sidebar.markdown("- Masquage sigmoid")
    st.sidebar.markdown("- Filtres : [16, 32, 64, 128]")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Format attendu**")
    st.sidebar.markdown("Fichier MP4 MUSDB18 (10 canaux)")
    st.sidebar.markdown("ch 0-1 : mix")
    st.sidebar.markdown("ch 2-3 : drums")
    st.sidebar.markdown("ch 4-5 : bass")
    st.sidebar.markdown("ch 6-7 : other")
    st.sidebar.markdown("ch 8-9 : vocals")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Metrique**")
    st.sidebar.markdown("SI-SDR calcule automatiquement depuis les stems embarques.")

    # ── Upload MP4 ────────────────────────────────────────────────────────────
    st.markdown("### 1. Charger un fichier MP4 MUSDB18")
    uploaded = st.file_uploader(
        "Deposez un fichier MP4 MUSDB18 (mix + 4 stems embarques)",
        type=["mp4"],
    )

    if uploaded is None:
        st.info("Uploadez un fichier MP4 MUSDB18 pour commencer.")
        st.markdown("---")
        st.markdown("#### Architecture du modele")
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/"
            "U-Net_architecture.svg/800px-U-Net_architecture.svg.png",
            caption="Architecture U-Net (Ronneberger et al., 2015)",
            width=700,
        )
        return

    # ── Extraction des stems depuis le MP4 ────────────────────────────────────
    with st.spinner("Extraction des stems depuis le fichier MP4..."):
        stems = extract_stems_from_mp4(uploaded)

    mix_audio = stems.get("mix")
    if mix_audio is None:
        st.error("Impossible d'extraire le mix depuis ce fichier.")
        return

    # ── Signal original ───────────────────────────────────────────────────────
    st.markdown("### 2. Signal original")
    st.audio(numpy_to_audio_bytes(mix_audio), format="audio/wav")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Duree :** {mix_audio.shape[1]/SR:.1f}s")
    with col2:
        st.markdown(f"**Canaux :** Stereo ({mix_audio.shape[0]}ch)")
    with col3:
        stems_found = [k for k in stems if k != "mix"]
        st.markdown(f"**Stems extraits :** {len(stems_found)}/4")

    with st.expander("Ecouter les stems de reference (ground truth)", expanded=False):
        ref_cols = st.columns(4)
        for i, src in enumerate(["vocals", "drums", "bass", "other"]):
            with ref_cols[i]:
                st.markdown(f"**{src}**")
                if src in stems:
                    st.audio(numpy_to_audio_bytes(stems[src]), format="audio/wav")
                else:
                    st.caption("non disponible")

    with st.expander("Spectrogramme du mix", expanded=False):
        fig = plot_spectrogram(mix_audio, "Mix — Spectrogramme de magnitude")
        st.pyplot(fig)
        plt.close(fig)

    # ── Séparation ────────────────────────────────────────────────────────────
    st.markdown("### 3. Separation de sources")

    if st.button("Lancer la separation", type="primary"):

        sources_to_process = (
            ["vocals", "drums", "bass", "other"]
            if source_choice == "all (4 sources)"
            else [source_choice]
        )

        for src in sources_to_process:
            st.markdown(f"#### {src.capitalize()}")

            model_path = Path(model_dir) / f"best_model_{src}.pt"

            # Reference directement extraite du MP4
            ref_audio = stems.get(src)

            # ---- Mini U-Net ------------------------------------------------
            unet_estimated = None
            if not model_path.exists():
                st.warning(
                    f"Modele pour `{src}` non trouve : {model_path}. "
                    f"Lancez : `python train.py --source {src} --musdb_root /path/to/musdb18`"
                )
            else:
                with st.spinner(f"Mini U-Net — separation : {src}..."):
                    model, device = load_unet_model(str(model_path))
                    unet_estimated = separate_unet(model, mix_audio, device)

            # ---- Open-Unmix -----------------------------------------------
            umx_estimated = None
            if compare_umx:
                # Le modele est charge une seule fois pour les 4 sources (requis par l'EM interne)
                separator = load_openunmix_model()
                if separator is None:
                    st.warning(
                        "Open-Unmix non installe. Lancez : `pip install openunmix`"
                    )
                else:
                    with st.spinner(f"Open-Unmix — separation : {src}..."):
                        umx_estimated = separate_openunmix(separator, mix_audio, src)

            # ---- Affichage résultats ---------------------------------------
            def _show_result(estimated, label, ref):
                st.markdown(f'<p class="compare-label">{label}</p>',
                            unsafe_allow_html=True)
                st.audio(numpy_to_audio_bytes(estimated), format="audio/wav")
                if ref is not None:
                    min_len = min(estimated.shape[1], ref.shape[1])
                    score = si_sdr(ref[:, :min_len], estimated[:, :min_len])
                    st.metric("SI-SDR", f"{score:.2f} dB")
                else:
                    st.caption("Stem de reference non disponible pour le SI-SDR.")
                fig = plot_spectrogram(estimated, f"{label} — {src}")
                st.pyplot(fig)
                plt.close(fig)

            if unet_estimated is not None and umx_estimated is not None:
                col_unet, col_umx = st.columns(2)
                with col_unet:
                    _show_result(unet_estimated, "Mini U-Net (entraine)", ref_audio)
                with col_umx:
                    _show_result(umx_estimated, "Open-Unmix UMX-L", ref_audio)

            elif unet_estimated is not None:
                col_audio, col_spec = st.columns([1, 2])
                with col_audio:
                    st.markdown('<p class="compare-label">Mini U-Net</p>',
                                unsafe_allow_html=True)
                    st.audio(numpy_to_audio_bytes(unet_estimated), format="audio/wav")
                    if ref_audio is not None:
                        min_len = min(unet_estimated.shape[1], ref_audio.shape[1])
                        score = si_sdr(ref_audio[:, :min_len], unet_estimated[:, :min_len])
                        st.metric("SI-SDR", f"{score:.2f} dB")
                    else:
                        st.caption("Stem de reference non disponible.")
                with col_spec:
                    fig = plot_spectrogram(unet_estimated, f"{src} — Spectrogramme estime")
                    st.pyplot(fig)
                    plt.close(fig)

        st.success("Separation terminee.")
        st.balloons()

    # ── A propos ──────────────────────────────────────────────────────────────
    with st.expander("A propos du modele"):
        st.markdown("""
        **Mini U-Net pour la séparation de sources audio**

        Version allégée du U-Net de Spleeter (Deezer Research, 2020),
        réimplémentée en PyTorch et entraînée from scratch sur MUSDB18.

        | Parametre | Valeur |
        |-----------|--------|
        | Architecture | U-Net Conv2D |
        | Filtres | [16, 32, 64, 128] |
        | STFT | n_fft=4096, hop=1024 |
        | Loss | L1 |
        | Optimizer | Adam |
        | Dataset | MUSDB18 (subset réduit) |

        **Comparaison — Open-Unmix (UMX-L)**

        Open-Unmix est le modèle de référence open-source pour la séparation audio.
        Entraîné sur MUSDB18 complet, il sert de baseline de référence.
        Installation : `pip install openunmix`

        **Metrique — SI-SDR**

        Le SI-SDR (Scale-Invariant SDR) mesure la qualité de la séparation en dB,
        indépendamment d'un facteur d'échelle global. Plus la valeur est élevée, meilleure
        est la séparation. Nécessite un signal de référence (stem isolé du même morceau).

        **Pipeline de traitement :**
        1. Audio → STFT → spectrogramme de magnitude
        2. U-Net → estimation du masque de Wiener (sigmoid)
        3. Masque × spectrogramme mix → spectrogramme source
        4. ISTFT (phase du mix) → signal audio reconstruit
        """)


if __name__ == "__main__":
    main()