import os
import librosa
import numpy as np
import soundfile as sf
import webrtcvad
import scipy.signal
from librosa import pyin
from typing import Tuple

# ==== CONFIG ====
INPUT_ROOT = "Train Set"
OUTPUT_ROOT = "Processed_Audio"
LABELS = ["bonafide", "spoofed"]
OUTPUT_TYPES = ["full_audio", "speech_only", "voiced_only", "unvoiced_only"]
SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
PYIN_FRAME_LENGTH = 2048
HOP_LENGTH = 160


# ==== Setup Output Folders ====
def setup_output_dirs():
    for out_type in OUTPUT_TYPES:
        for label in LABELS:
            os.makedirs(os.path.join(OUTPUT_ROOT, out_type, label), exist_ok=True)


# ==== WebRTC VAD-based speech mask ====
def get_speech_mask(audio: np.ndarray, sr: int, aggressiveness: int = 2) -> np.ndarray:
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * VAD_FRAME_MS / 1000)
    n_frames = len(audio) // frame_size
    mask = np.zeros(len(audio), dtype=bool)

    for i in range(n_frames):
        start = i * frame_size
        end = start + frame_size
        frame = audio[start:end]
        if len(frame) < frame_size:
            continue
        pcm = (frame * 32768).astype(np.int16).tobytes()
        if vad.is_speech(pcm, sr):
            mask[start:end] = True

    return mask


# ==== Voicing Detection ====
def get_voiced_unvoiced(audio: np.ndarray, sr: int, speech_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    f0, voiced_flag, _ = pyin(audio, fmin=librosa.note_to_hz('C2'),
                              fmax=librosa.note_to_hz('C7'),
                              frame_length=PYIN_FRAME_LENGTH,
                              sr=sr, hop_length=HOP_LENGTH)

    voiced_interp = np.repeat(voiced_flag.astype(float), HOP_LENGTH)[:len(audio)]
    voiced_interp = scipy.signal.convolve(voiced_interp, np.hamming(10), mode='same')
    voiced_interp = np.clip(voiced_interp, 0, 1)

    y_voiced = audio * speech_mask * voiced_interp
    y_unvoiced = audio * speech_mask * (1 - voiced_interp)
    return y_voiced, y_unvoiced


# ==== Main Processing ====
def process_file(filepath: str, label: str):
    try:
        filename = os.path.basename(filepath)
        if not filename.lower().endswith(".flac"):
            return

        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        speech_mask = get_speech_mask(audio, sr)

        y_full = audio
        y_speech = audio * speech_mask
        y_voiced, y_unvoiced = get_voiced_unvoiced(audio, sr, speech_mask)

        base_name = filename.replace(".flac", ".wav")
        sf.write(os.path.join(OUTPUT_ROOT, "full_audio", label, base_name), y_full, sr)
        sf.write(os.path.join(OUTPUT_ROOT, "speech_only", label, base_name), y_speech, sr)
        sf.write(os.path.join(OUTPUT_ROOT, "voiced_only", label, base_name), y_voiced, sr)
        sf.write(os.path.join(OUTPUT_ROOT, "unvoiced_only", label, base_name), y_unvoiced, sr)

        print(f"✅ Processed: {label}/{filename}")
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")


def main():
    setup_output_dirs()
    for label in LABELS:
        input_dir = os.path.join(INPUT_ROOT, label)
        for file in os.listdir(input_dir):
            process_file(os.path.join(input_dir, file), label)


if __name__ == "__main__":
    main()
