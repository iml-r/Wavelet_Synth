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

def adjust_spectral_tilt(waveform, tilt_db_per_octave=-6.0):
    """
    Applies a spectral tilt to a single-period waveform such that lower frequencies are emphasized.

    Parameters:
        waveform (np.ndarray): A 1D array representing a single-period waveform.
        tilt_db_per_octave (float): The amount of tilt in decibels per octave (negative for low-frequency boost).

    Returns:
        np.ndarray: The waveform with modified spectral tilt.
    """
    N = len(waveform)
    freqs = np.fft.rfftfreq(N)
    spectrum = np.fft.rfft(waveform)

    # Avoid division by zero at DC
    freqs[0] = freqs[1] if N > 1 else 1.0

    # Calculate gain factors for tilt: G(f) = 10^(tilt * log2(f)) = f^(tilt / (20 * log10(2)))
    tilt_factor = tilt_db_per_octave / 6.0  # 6 dB per octave = x2 gain per frequency doubling
    gain = np.power(freqs, tilt_factor)
    gain[0] = 1.0  # Keep DC unchanged

    # Apply gain
    tilted_spectrum = spectrum * gain

    # Inverse FFT to get back to time domain
    tilted_waveform = np.fft.irfft(tilted_spectrum, n=N)

    return tilted_waveform

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
    if compress:
        waveform = compress_spectral_range(waveform, strength=0.7)

    waveform = remove_dc(waveform)
    waveform = normalize(waveform)
    waveform = match_slopes(waveform)
    waveform = match_endpoints(waveform)

    return waveform