import os
from pyo import *
import numpy as np
import pandas as pd

device_index = 1

# Boot server with explicit MIDI device
s = Server()
s.setMidiInputDevice(device_index)
s.boot()
s.start()

def safe_index(index:int , minimum:int = 0, maximum: int = 99):
    return np.min([np.max([minimum, index]), maximum])

# --- 1. Load Wavetables ---
base_oscillator_dict = {}
for base_oscillator in os.listdir("wavetable_bank"):
    mother_wavelet_dict = {}
    for mother_wavelet in os.listdir(f"wavetable_bank/{base_oscillator}/"):
        one_hot_lst = []
        for one_hot_value in sorted(os.listdir(f"wavetable_bank/{base_oscillator}/{mother_wavelet}/")):
            one_hot_value = int(one_hot_value)
            df_path = f"wavetable_bank/{base_oscillator}/{mother_wavelet}/{one_hot_value}/wavetables.csv"
            wavebank_df = pd.read_csv(df_path, sep=",", header=0)

            base_periods_lst = []
            for _, row in wavebank_df.iterrows():
                wave_array = np.array(row)
                tbl = DataTable(size=len(wave_array))
                tbl.replace(wave_array.tolist())
                base_periods_lst.append(tbl)

            one_hot_lst.append(base_periods_lst)
        mother_wavelet_dict[mother_wavelet] = one_hot_lst
    base_oscillator_dict[base_oscillator] = mother_wavelet_dict
print("Wavetables loaded")

# --- 2. Hardcoded Table Selector Paths (for now) ---
base_oscillator = "sin"
mother_wavelet = "haar"
table_bank = base_oscillator_dict[base_oscillator][mother_wavelet]  # shape: [one_hot][base_period]
print("Table bank selected")

# --- 3. MIDI Controls for Wavetable Selection ---
# Knob 1: One-hot selector (0-3)
one_hot_cc = Midictl(ctlnumber=19, minscale=0, maxscale=3, init=0)
# Knob 2: Base period selector (0-99)
base_period_cc = Midictl(ctlnumber=20, minscale=0, maxscale=98, init=0)
print("Controls for wavetable selection defined")

# --- 4. MIDI ADSR Controls ---
attack_ctrl = Midictl(ctlnumber=73, minscale=0.001, maxscale=2, init=0.01)
decay_ctrl = Midictl(ctlnumber=75, minscale=0.001, maxscale=2, init=0.2)
sustain_ctrl = Midictl(ctlnumber=30, minscale=0.0, maxscale=1.0, init=0.7)
release_ctrl = Midictl(ctlnumber=72, minscale=0.001, maxscale=3.0, init=0.5)
print("foo_ctrl defined")

# Smooth MIDI control values
attack_sig = SigTo(value=attack_ctrl.get(), time=0.05)
decay_sig = SigTo(value=decay_ctrl.get(), time=0.05)
sustain_sig = SigTo(value=sustain_ctrl.get(), time=0.05)
release_sig = SigTo(value=release_ctrl.get(), time=0.05)

# attack_sig = SigTo(value=0.01, time=0.05)
# decay_sig = SigTo(value=0.2, time=0.05)
# sustain_sig = SigTo(value=0.7, time=0.05)
# release_sig = SigTo(value=0.5, time=0.05)

print("foo_sig defined")

# Update function polls MIDI and updates smooth signals
def update_adsr_params():
    attack_sig.value = attack_ctrl.get()
    decay_sig.value = decay_ctrl.get()
    sustain_sig.value = sustain_ctrl.get()
    release_sig.value = release_ctrl.get()

print("update_adsr_params defined")


adsr_pat = Pattern(update_adsr_params, 0.05)
adsr_pat.play()

# Helper to create ADSR curve points
def create_adsr_points(a, d, s, r, dur=4):
    attack_time = a
    decay_time = d
    release_time = r
    sustain_time = max(dur - (a + d + r), 0)
    return [
        (0, 0),
        (attack_time, 1),
        (attack_time + decay_time, s),
        (attack_time + decay_time + sustain_time, s),
        (attack_time + decay_time + sustain_time + release_time, 0),
    ]

print("ADSR defined")

# Create and return a Linseg envelope from current sig values
def make_env():
    points = create_adsr_points(
        attack_sig.value,
        decay_sig.value,
        sustain_sig.value,
        release_sig.value,
    )
    print("ADS envelope points:", points)

    times, vals = zip(*points)
    durations = [times[i+1] - times[i] for i in range(len(times)-1)]
    print("Durations:", durations)
    print("Values:", vals)

    durations = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    linseg_list = [(vals[i], durations[i]) for i in range(len(durations))]
    linseg_list.append((vals[-1], 0))  # or just (vals[-1],) if allowed

    print("Linseg list:", linseg_list)

    return Linseg(list=linseg_list, loop=False, mul=0.3)

print("make_env defined")
# This will hold the current envelope object
env = make_env()
print("make_env called")


print("Controls for ADSR defined, envelope object created")

# --- 5. Low-pass filter cutoff control on MIDI CC 74 ---
lp_cutoff_ctrl = Midictl(ctlnumber=74, minscale=20, maxscale=12000, init=5000)  # cutoff freq range
lp_cutoff_sig = SigTo(value=lp_cutoff_ctrl.get(), time=0.05)  # smooth cutoff freq

def update_lp_cutoff():
    lp_cutoff_sig.value = lp_cutoff_ctrl.get()

lp_cutoff_pat = Pattern(update_lp_cutoff, 0.05)
lp_cutoff_pat.play()
print("High-pass filter cutoff control defined")

# --- 6. Note Event Handling ---
notein = Notein(poly=16)
pitch = notein['pitch']
velocity = notein['velocity']
note_on = notein['trigon']
note_off = notein['trigoff']

print(notein)
print(f"note_on is: {note_on}")
print(f"type(note_on): {type(note_on)}")

# Convert pitch to freq and apply frequency envelope modulation
base_freq = MToF(pitch)

# Frequency envelope (modulates pitch at attack, then decays)
freq_env_amt = Midictl(ctlnumber=74, minscale=0, maxscale=12, init=0)
freq_env = Adsr(attack=0.01, decay=0.3, sustain=0.0, release=0.2, dur=1, mul=freq_env_amt)
freq_mod = base_freq * Pow(2, freq_env / 12)  # semitone-based pitch modulation
print("Note handling defined")

# --- 7. ADSR Envelope ---
env = Adsr(
    attack=attack_sig.value,
    decay=decay_sig.value,
    sustain=sustain_sig.value,
    release=release_sig.value,
    dur=4,
    mul=0.3
)
print("ADSR envelope devined")

# --- 8. Table Selection Logic + Play Note ---

current_osc = None
current_env = None
current_lp_filter = None

def play_note():
    global current_osc, current_env, current_lp_filter

    # Stop previous sounds
    if current_env is not None:
        current_env.stop()
    if current_osc is not None:
        current_osc.stop()
    if current_lp_filter is not None:
        current_lp_filter.stop()

    # ADSR params
    a = attack_sig.value
    d = decay_sig.value
    s_val = sustain_sig.value
    r = release_sig.value

    # Create new ADSR envelope and play
    current_env = Adsr(
        attack=attack_sig,
        decay=decay_sig,
        sustain=sustain_sig,
        release=release_sig,
    )
    current_env.play()

    # Get frequency for current pitch
    freq_val = MToF(pitch)  # PyoObject

    # Wavetable selection
    oh_idx = int(one_hot_cc.get())
    bp_idx = int(base_period_cc.get())
    try:
        tbl = table_bank[oh_idx][bp_idx]
    except IndexError:
        print(f"ERROR: Wavetable index out of range: one_hot_cc={oh_idx}, base_period_cc={bp_idx}")
        return

    # Create oscillator from wavetable and freq, apply envelope
    current_osc = Osc(table=tbl, freq=freq_val, mul=current_env)
    current_osc.out()

    # Apply low-pass filter on oscillator output
    cutoff = lp_cutoff_ctrl.get()
    current_lp_filter = ButLP(current_osc, freq=cutoff).out()

def update_filter_cutoff():
    if current_lp_filter is not None:
        cutoff = lp_cutoff_ctrl.get()
        current_lp_filter.freq = cutoff

filter_update_pat = Pattern(update_filter_cutoff, 0.05)
filter_update_pat.play()

def note_on_event():
    print("Note ON")
    play_note()

def note_off_event():
    print("Note OFF")
    if current_env is not None:
        current_env.stop()  # triggers release

TrigFunc(note_on, note_on_event)
TrigFunc(note_off, note_off_event)
print("Trig_func defined")

# --- GUI if needed ---
s.gui(locals())
print("GUI fired up")