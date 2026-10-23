import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

def load_wav(path, target_sr=16000):
    sr, raw = wavfile.read(path)
    if np.issubdtype(raw.dtype, np.integer):
        info = np.iinfo(raw.dtype)
        scale = float(max(abs(info.min), info.max))
        x = raw.astype(np.float32) / scale
    else:
        x = raw.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if target_sr and sr != target_sr:
        g = gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g)
        sr = target_sr
    return int(sr), x.astype(np.float32)

def extract_features(x, sr):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        raise ValueError("Empty audio")
    rms = float(np.sqrt(np.mean(x*x) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x)))) if x.size > 1 else 0.0)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, d=1.0/sr)
    peak_idx = int(np.argmax(spec[1:])+1) if spec.size > 1 else 0
    dominant = float(freqs[peak_idx]) if freqs.size else 0.0
    energy = float(np.sum(spec**2) / max(x.size, 1))
    centroid = float(np.sum(freqs*spec) / (np.sum(spec)+1e-12))
    return {"rms":rms, "zcr":zcr, "dominant_frequency":dominant,
            "spectral_energy":energy, "spectral_centroid":centroid}
