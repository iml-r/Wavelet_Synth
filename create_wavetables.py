import os

import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile

from scipy.signal import find_peaks, resample
from scipy.interpolate import interp1d
from PIL import Image
import io
from tqdm import tqdm
from wavetable_preprocessing import process_waveform, remove_dc, normalize, match_endpoints, match_slopes
from scipy.io.wavfile import write as write_wav
from scipy import signal

good_mother_wavelets = ["haar", "db17"]

def dirty_plot(a: np.ndarray):
    plt.plot([*range(len(a))], a)
    plt.show()


def autocorrelation(x: np.ndarray) -> np.ndarray:
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]



def one_hot_reconstruction(oscillator_signal: np.ndarray,
                           n=3,
                           mother_wavelet="haar") -> np.ndarray:
    wavelets = pywt.wavedec(oscillator_signal,
                            wavelet=mother_wavelet,
                            level=4,
                            mode='periodization')
    wavelets = tuple([level if (index == n) else (level * 0) for index, level in enumerate(wavelets)])
    reconstructed_signal = pywt.waverec(wavelets,
                                        wavelet=mother_wavelet,
                                        mode="periodization")
    return reconstructed_signal


def find_period_index(signal: np.ndarray) -> int:
    autocorrelated_signal = autocorrelation(signal)
    peak_indices, _ = find_peaks(autocorrelated_signal)

    if len(peak_indices) == 0:
        # Fallback: return 1 or a default lag
        return len(signal)-1  # or: return len(signal) // 2

    highest_peak_index = np.argmax(autocorrelated_signal[peak_indices])
    return peak_indices[highest_peak_index]


def align_zero_crossing(waveform):
    # Find zero-crossings (with positive slope)
    zero_crossings = np.where(np.diff(np.sign(waveform)) > 0)[0]
    if len(zero_crossings) < 2:
        return waveform  # fallback

    start = zero_crossings[0]
    end = zero_crossings[1]
    aligned = waveform[start:end]
    return np.interp(np.linspace(0, len(aligned) - 1, len(waveform)), np.arange(len(aligned)), aligned)


def single_wave(signal: np.ndarray,
                samples=2048) -> np.ndarray:
    period_index = find_period_index(signal)
    single_period = signal[:period_index]
    resampled = resample(single_period, num=samples)

    return resampled


def oscillator(periods, resolution=2048, waveform='sine', seed=None):
    """
    Generate a waveform oscillator.

    Parameters:
    - periods: number of cycles in the output
    - resolution: number of samples
    - waveform: 'sine', 'square', 'triangle', 'saw', 'noise', 'weierstrass'
    - seed: optional random seed for reproducibility (only affects noise)

    Returns:
    - output_signal: numpy array of the waveform
    """
    t = np.linspace(0, 1, resolution, endpoint=False)  # normalized time [0,1)
    freq = periods

    if waveform == 'sine':
        output_signal = np.sin(2 * np.pi * freq * t)

    elif waveform == 'square':
        output_signal = signal.square(2 * np.pi * freq * t)

    elif waveform == 'triangle':
        output_signal = signal.sawtooth(2 * np.pi * freq * t, width=0.5)  # 0.5 = triangle

    elif waveform == 'saw':
        output_signal = signal.sawtooth(2 * np.pi * freq * t)

    elif waveform == 'noise':
        rng = np.random.default_rng(seed)
        output_signal = rng.uniform(low=-1.0, high=1.0, size=resolution)

    elif waveform == 'weierstrass':
        # Basic Weierstrass function implementation
        # Parameters can be adjusted for more/less roughness
        a = 0.5
        b = 3
        n_terms = 30  # more = more detail
        output_signal = np.zeros_like(t)
        for n in range(n_terms):
            output_signal += a**n * np.cos(b**n * 2 * np.pi * t * freq)

        # Normalize to [-1, 1]
        output_signal /= np.max(np.abs(output_signal))

    else:
        raise ValueError(f"Unsupported waveform: {waveform}")

    return output_signal


def add_fundamental_contrast(waveform, contrast_db=6):
    N = len(waveform)
    fft = np.fft.fft(waveform)

    # Harmonic bins (assuming bin 1 = fundamental)
    fundamental_bin = 1
    harmonics_to_attenuate = [2, 3, 4, 5, 6, 7]

    # Contrast amount in linear scale
    contrast_ratio = 10 ** (contrast_db / 20)

    # Boost fundamental
    fft[fundamental_bin] *= contrast_ratio
    fft[-fundamental_bin] *= contrast_ratio  # mirror freq

    # Attenuate harmonics just above
    for h in harmonics_to_attenuate:
        if h < N // 2:
            fft[h] /= contrast_ratio
            fft[-h] /= contrast_ratio

    # Return normalized time-domain waveform
    result = np.fft.ifft(fft).real
    return result / np.max(np.abs(result))

def wavelet_synthesis(periods: int, detail_level: int, mother_wavelet="haar", compress=True, waveform="sine"):
    signal = oscillator(periods, waveform=waveform)
    raw_synthesis = one_hot_reconstruction(signal, n=detail_level, mother_wavelet=mother_wavelet)
    single_synthesis_period = single_wave(raw_synthesis)
    single_synthesis_period = process_waveform(single_synthesis_period, compress=compress)
    #single_synthesis_period = add_fundamental_contrast(single_synthesis_period,contrast_db=6)

    return single_synthesis_period


##FREQ SHOULD NOT BE HIGHER THAN 100
# I don't know why

freq = 2
t = np.linspace(start=0, stop=2 * np.pi, num=4096)
sine_wave = np.sin(t * freq)
print(find_period_index(sine_wave))

# dirty_plot(sine_wave)
# dirty_plot(one_hot_reconstruction(sine_wave, n=2))
# dirty_plot(single_wave(sine_wave))
dirty_plot(single_wave(one_hot_reconstruction(sine_wave, n=0)))

wavelets = pywt.wavedec(sine_wave,
                        wavelet='db17',
                        level=4,
                        mode='periodization')

def mean_abs(x: np.ndarray,y:np.ndarray) -> np.float32:
    x, y = np.array(x), np.array(y)
    return np.mean(np.abs(x-y))

def sort_wavetable(df: pd.DataFrame) -> pd.DataFrame:
    wavelst = []
    for _, row in df.iterrows():
        wavelst.append(np.array(row))

    key_function = lambda x: np.mean([mean_abs(x, y) for y in wavelst])
    keys = df.apply(key_function, axis=1)
    return df.iloc[keys.argsort()].reset_index(drop=True)

def render_plot_to_image(a: np.ndarray) -> Image.Image:
    """Render a plot to a PIL image, from a 1D numpy array."""
    fig, ax = plt.subplots()
    ax.plot(range(len(a)), a)
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    plt.close(fig)
    return img


# Generate your frames
for waveform in tqdm(['sine', 'square', 'triangle', 'saw', 'noise', 'weierstrass']):
    for detail_level in range(4):
        for mother_wavelet in ["haar", "dmey", "bior1.1", "bior2.8", "db1", "db7", "db38", "sym2", "sym10"]:
            frames = []
            jump = []
            wavetable_lst = []
            for periods in range(2, 100):
                x = wavelet_synthesis(periods=periods,
                                      detail_level=detail_level,
                                      mother_wavelet=mother_wavelet,
                                      waveform=waveform)
                wavetable_lst.append(x)

                frame = render_plot_to_image(x)
                frames.append(frame)

            # Save as GIF
            path = f"wavetable_bank/{waveform}/{mother_wavelet}/{detail_level}/"
            os.makedirs(path, exist_ok=True)
            frames[0].save(
                path + f"output{detail_level}.gif",
                save_all=True,
                append_images=frames[1:],
                duration=100,  # milliseconds per frame
                loop=0  # 0 = loop forever
            )

            # Save as df
            wavetable = np.stack(wavetable_lst)
            wavetable_df = pd.DataFrame(wavetable)
            #wavetable_df = sort_wavetable(wavetable_df)
            wavetable_df.to_csv(path+"wavetables.csv", index=False, index_label=False)

            # Save as wav
            rowlst = []
            for index,row in wavetable_df.iterrows():
                rowlst.append(np.array(row))

            samplerate = 44100
            fs = 100
            write_wav(path+f"wavetable_{waveform}_{mother_wavelet}_{detail_level}.wav", data=np.concat(rowlst).astype(np.float32), rate=44100)