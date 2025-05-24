import numpy as np

from scipy.signal import resample, normalize

def match_endpoints(waveform):
    """Linearly morph the waveform to make start and end values match."""
    diff = waveform[-1] - waveform[0]
    ramp = np.linspace(0, -diff, len(waveform))
    return waveform + ramp

def match_slopes(waveform):
    """Smooth start and end slopes to match using overlap-add window."""
    # Estimate slopes
    slope_start = waveform[1] - waveform[0]
    slope_end = waveform[-1] - waveform[-2]
    slope_diff = slope_end - slope_start

    # Apply a cosine ramp correction
    ramp = np.cos(np.linspace(0, np.pi, len(waveform)))
    correction = slope_diff * (1 - ramp) * 0.5  # bell-shaped
    return waveform - correction

def remove_dc(waveform):
    return waveform - np.mean(waveform)

def normalize(waveform):
    peak = np.max(np.abs(waveform))
    return waveform / peak if peak != 0 else waveform

def resample_to_length(waveform, target_len=2048):
    return resample(waveform, target_len)

def compress_spectral_range(waveform, strength=0.7):
    fft = np.fft.rfft(waveform)
    mag = np.abs(fft)
    phase = np.angle(fft)
    mag_compressed = mag ** strength
    fft_new = mag_compressed * np.exp(1j * phase)
    return np.fft.irfft(fft_new, n=len(waveform))

def soft_loop_fix(waveform):
    diff = waveform[-1] - waveform[0]
    ramp = np.linspace(0, -diff, len(waveform))
    return waveform + ramp

def force_loop_continuity(waveform):
    """Correct both value and slope mismatch at loop point."""
    N = len(waveform)

    # Match end value to start
    delta_value = waveform[-1] - waveform[0]
    value_ramp = np.linspace(0, -delta_value, N)
    waveform = waveform + value_ramp

    # Match slope (1st derivative)
    slope_start = waveform[1] - waveform[0]
    slope_end = waveform[-1] - waveform[-2]
    delta_slope = slope_end - slope_start

    slope_ramp = np.cos(np.linspace(0, np.pi, N)) * delta_slope * 0.5
    waveform = waveform - slope_ramp

    return waveform
def process_waveform(waveform, compress=False):
    waveform = remove_dc(waveform)
    waveform = normalize(waveform)
    waveform = match_endpoints(waveform)

    if compress:
        waveform = compress_spectral_range(waveform, strength=0.7)
    return waveform