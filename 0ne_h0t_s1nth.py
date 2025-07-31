import os
from pyo import *
import numpy as np
import pandas as pd
from pythonosc import dispatcher, osc_server
import threading

# --- PYO SERVER SETUP ---
s = Server(buffersize=16, nchnls=2, duplex=1, sr=44100)
s.boot()
s.start()

# --- MIDI CONTROL UTILS ---
def safe_index(index:int , minimum:int = 0, maximum: int = 98):
    return np.min([np.max([minimum, index]), maximum])

def midi_cc_to_index(cc_val, target_len):
    # Map 1–128 to 0–(target_len - 1)
    idx = int((cc_val - 1) / 127 * (target_len - 1))
    return safe_index(idx, 0, target_len - 1)

# --- LOAD WAVETABLES ---
base_oscillator_dict = {}
for base_oscillator in os.listdir("wavetable_bank"):
    mother_wavelet_dict = {}
    for mother_wavelet in os.listdir(f"wavetable_bank/{base_oscillator}/"):
        one_hot_lst = []
        for one_hot_value in sorted(os.listdir(f"wavetable_bank/{base_oscillator}/{mother_wavelet}/")):
            one_hot_value = int(one_hot_value)
            df_path = f"wavetable_bank/{base_oscillator}/{mother_wavelet}/{one_hot_value}/wavetables.csv"
            wavebank_df = pd.read_csv(df_path)

            base_periods_lst = []
            for _, row in wavebank_df.iterrows():
                wave_array = np.array(row)
                tbl = DataTable(size=len(wave_array))
                tbl.replace(wave_array.tolist())
                base_periods_lst.append(tbl)

            one_hot_lst.append(base_periods_lst)
        mother_wavelet_dict[mother_wavelet] = one_hot_lst
    base_oscillator_dict[base_oscillator] = mother_wavelet_dict

print("Wavetables loaded.")

# --- FIXED PATHS ---
base_oscillator = "weierstrass"  # fixed for now
one_hot_idx = 0           # fixed for now

# --- MIDI CC CONTROL STATE ---
raw_mother_cc_val = 1
raw_baseperiod_cc_val = 1
lp_cutoff_val = 5000

# --- SIGS FOR ADSR AND FILTER ---
attack_sig = Sig(0.01)
decay_sig = Sig(0.2)
sustain_sig = Sig(0.7)
release_sig = Sig(0.5)

lp_cutoff_sig = SigTo(value=lp_cutoff_val, time=0.05)

def update_lp_cutoff():
    lp_cutoff_sig.value = lp_cutoff_val

Pattern(update_lp_cutoff, 0.05).play()

# --- GLOBAL STATE ---
current_osc = None
current_env = None
current_lp_filter = None

freq = 440

# --- TABLE SELECTOR FUNCTION ---
def get_current_table():
    mother_wavelet_list = sorted(base_oscillator_dict[base_oscillator].keys())
    mw_idx = midi_cc_to_index(raw_mother_cc_val, len(mother_wavelet_list))
    mother_wavelet = mother_wavelet_list[mw_idx]

    bp_idx = midi_cc_to_index(raw_baseperiod_cc_val, 99)

    try:
        table = base_oscillator_dict[base_oscillator][mother_wavelet][one_hot_idx][bp_idx]
        return table, mother_wavelet, bp_idx
    except Exception as e:
        print(f"Error selecting wavetable: {e}")
        return None, mother_wavelet, bp_idx

# --- PLAY / STOP NOTE FUNCTIONS ---
def play_note_from_osc(note, velocity):
    global current_osc, current_env, current_lp_filter, freq

    freq = MToF(Sig(note))

    tbl, mw_name, bp_idx = get_current_table()
    if tbl is None:
        print("Failed to retrieve wavetable.")
        return

    print(f"Playing: {base_oscillator}/{mw_name} [bp {bp_idx}]")

    # If already playing a note, stop envelope & oscillator cleanly first
    if current_env is not None:
        current_env.stop()
    if current_osc is not None:
        current_osc.stop()
    if current_lp_filter is not None:
        current_lp_filter.stop()

    current_env = Adsr(
        attack=attack_sig.value,
        decay=decay_sig.value,
        sustain=sustain_sig.value,
        release=release_sig.value,
        dur=0,             # zero means envelope length depends on ADSR, no forced duration
        mul=velocity / 127 * 0.3
    ).play()

    current_osc = Osc(table=tbl, freq=freq, mul=current_env)
    current_lp_filter = ButLP(current_osc, freq=lp_cutoff_sig)
    current_lp_filter.out()


def stop_note_from_osc():
    global current_env, current_osc, current_lp_filter

    if current_env is not None:
        current_env.stop()  # triggers release phase if dur=0 or sustain phase

        release_time = release_sig.value  # get release duration

        def cleanup():
            global current_env, current_osc, current_lp_filter
            if current_env is not None:
                current_env.stop()
                current_env = None
            if current_osc is not None:
                current_osc.stop()
                current_osc = None
            if current_lp_filter is not None:
                current_lp_filter.stop()
                current_lp_filter = None

        CallAfter(cleanup, time=release_time).play()
# --- UPDATE CONTROLS FROM CC ---

def update_osc_table_live():
    global current_osc
    tbl, _, _ = get_current_table()
    if tbl is not None and current_osc is not None:
        current_osc.setTable(tbl)  # <-- change wavetable on the fly

def update_controls_from_cc(control, value):
    global raw_mother_cc_val, raw_baseperiod_cc_val, lp_cutoff_val
    if control == 81:
        raw_mother_cc_val = value
        print(f"Updated mother_wavelet CC to {value}")
        update_osc_table_live()  # update wavetable while note is playing
    elif control == 82:
        raw_baseperiod_cc_val = value
        print(f"Updated base_period CC to {value}")
        update_osc_table_live()  # update wavetable while note is playing
    elif control == 74:
        lp_cutoff_val = np.interp(value, [0, 127], [20, 12000])
        print(f"Updated LP filter cutoff to {lp_cutoff_val:.1f} Hz")
# --- OSC HANDLERS ---
def osc_note_on(addr, args, note, velocity):
    play_note_from_osc(note, velocity)

def osc_note_off(addr, args, note):
    stop_note_from_osc()

def osc_cc(addr, args, control, value):
    update_controls_from_cc(control, value)

# --- SETUP OSC SERVER ---
disp = dispatcher.Dispatcher()
disp.map("/note_on", osc_note_on, "note_on")
disp.map("/note_off", osc_note_off, "note_off")
disp.map("/cc", osc_cc, "cc")

def start_osc_server():
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 12345), disp)
    print("Starting OSC server on 127.0.0.1:12345")
    server.serve_forever()

threading.Thread(target=start_osc_server, daemon=True).start()


# --- LIVE WAVETABLE SWEEPING ---
def update_table_during_note():
    global current_osc
    if current_osc is not None:
        tbl, _, _ = get_current_table()
        if tbl is not None:
            current_osc.setTable(tbl)

Pattern(update_table_during_note, time=0.05).play()

# --- START PYO GUI ---
s.gui(locals())
