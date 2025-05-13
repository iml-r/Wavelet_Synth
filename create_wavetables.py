import pandas as pd
import pywt
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, resample

from PIL import Image
import io

from tqdm import tqdm

def dirty_plot(a: np.ndarray):
    plt.plot([*range(len(a))], a)
    plt.show()

def autocorrelation(x: np.ndarray) -> np.ndarray:
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def one_hot_reconstruction(oscillator_signal: np.ndarray,
                           n=3,
                           mother_wavelet="haar") -> tuple:
    wavelets = pywt.wavedec(oscillator_signal,
                            wavelet=mother_wavelet,
                            level=4,
                            mode='periodization')
    wavelets = tuple([level if (index == n) else (level*0) for index,level in enumerate(wavelets)])
    reconstructed_signal = pywt.waverec(wavelets,
                                        wavelet=mother_wavelet,
                                        mode="periodization")
    return reconstructed_signal

def find_period_index(signal: np.ndarray) -> int:
    autocorrelated_signal = autocorrelation(signal)
    peak_indices, _ = find_peaks(autocorrelated_signal)
    highest_peak_index = np.argmax(autocorrelated_signal[peak_indices])
    return peak_indices[highest_peak_index]

def single_wave(signal: np.ndarray,
                samples=2048) -> np.ndarray:
    period_index = find_period_index(signal)
    single_period = signal[:period_index]
    resampled = resample(single_period, num=samples)

    return resampled


##FREQ SHOULD NOT BE HIGHER THAN 100
#I don't know why
for f in [2,5,10,20,30,40,50,60]:
    freq = f
    t = np.linspace(start=0, stop=2*np.pi, num=2048)
    sine_wave = np.sin(t*freq)
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
for detail_level in range(4):
    frames = []
    for f in range(2, 70):
        freq = f
        t = np.linspace(start=0, stop=2*np.pi, num=2048)
        sine_wave = np.sin(t * freq)

        # Replace with your actual signal logic
        x = single_wave(one_hot_reconstruction(sine_wave, n=detail_level))

        frame = render_plot_to_image(x)
        frames.append(frame)

    # Save as GIF
    frames[0].save(
        f"output{detail_level}.gif",
        save_all=True,
        append_images=frames[1:],
        duration=300,  # milliseconds per frame
        loop=0         # 0 = loop forever
    )