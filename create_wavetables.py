import os

import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, resample

from PIL import Image
import io

from tqdm import tqdm

from wavetable_preprocessing import process_waveform, remove_dc, normalize, match_endpoints, match_slopes


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


def oscillator(periods, resolution=4096):
    freq = periods
    t = np.linspace(start=0, stop=2 * np.pi, num=resolution)
    output_signal = np.sin(t * freq)

    return output_signal


def wavelet_synthesis(periods: int, detail_level: int, compress=False):
    signal = oscillator(periods)
    raw_synthesis = one_hot_reconstruction(signal, n=detail_level)
    single_synthesis_period = single_wave(raw_synthesis)
    single_synthesis_period = process_waveform(single_synthesis_period, compress=compress)

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
                        wavelet='haar',
                        level=4,
                        mode='periodization')


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
for detail_level in tqdm(range(4)):
    frames = []
    jump = []
    wavetable_lst = []
    for periods in range(2, 100):
        x = wavelet_synthesis(periods=periods, detail_level=detail_level)
        wavetable_lst.append(x)

        frame = render_plot_to_image(x)
        frames.append(frame)

    # Save as GIF
    path = f"wavetable_bank/sin/haar/{detail_level}/"
    os.makedirs(path, exist_ok=True)
    frames[0].save(
        path + f"output{detail_level}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,  # milliseconds per frame
        loop=0  # 0 = loop forever
    )

    wavetable = np.stack(wavetable_lst)
    wavetable_df = pd.DataFrame(wavetable)

    wavetable_df.to_csv(path+"wavetables.csv", index=False, index_label=False)
