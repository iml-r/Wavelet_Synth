import os
import time
import threading
import numpy as np
import pandas as pd
from pyo import *
import mido

# =======================
# Configuration & Setup
# =======================
MIDI_DEVICE_INDEX = 1
WAVETABLE_DIR = "wavetable_bank"

ports = mido.get_input_names()
print(f"Available MIDI ports: {ports}")


# =======================
# Utility Functions
# =======================

def safe_index(index: int, minimum: int = 0, maximum: int = 99) -> int:
    """Clamp index between minimum and maximum."""
    return max(minimum, min(index, maximum))


def load_wavetables(base_dir: str):
    """Load wavetable data into a nested dictionary."""
    wavetables = {}
    for base_osc in os.listdir(base_dir):
        mother_wavelets = {}
        base_osc_path = os.path.join(base_dir, base_osc)
        for mother_wavelet in os.listdir(base_osc_path):
            one_hot_list = []
            mw_path = os.path.join(base_osc_path, mother_wavelet)
            for one_hot_val in sorted(os.listdir(mw_path), key=int):
                df_path = os.path.join(mw_path, one_hot_val, "wavetables.csv")
                df = pd.read_csv(df_path)
                base_periods = []
                for _, row in df.iterrows():
                    wave_array = np.array(row)
                    tbl = DataTable(size=len(wave_array))
                    tbl.replace(wave_array.tolist())
                    base_periods.append(tbl)
                one_hot_list.append(base_periods)
            mother_wavelets[mother_wavelet] = one_hot_list
        wavetables[base_osc] = mother_wavelets
    return wavetables


def create_adsr_points(a, d, s, r, duration=4):
    sustain_time = max(duration - (a + d + r), 0)
    return [
        (0, 0),
        (a, 1),
        (a + d, s),
        (a + d + sustain_time, s),
        (a + d + sustain_time + r, 0),
    ]


def make_linseg_env(attack, decay, sustain, release):
    points = create_adsr_points(attack, decay, sustain, release)
    times, vals = zip(*points)
    durations = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    linseg_list = [(vals[i], durations[i]) for i in range(len(durations))]
    linseg_list.append((vals[-1], 0))
    return Linseg(list=linseg_list, loop=False, mul=0.3)

# =======================
# Voice Class
# =======================

class Voice:
    def __init__(self, freq, velocity, tbl, adsr_params, lp_cutoff, resonance):
        a, d, s, r = adsr_params
        self.env = Adsr(attack=a, decay=d, sustain=s, release=r)
        self.env.play()
        self.osc = Osc(table=tbl, freq=freq, mul=self.env)
        self.res_sig = SigTo(value=resonance, time=0.05)
        self.cutoff_sig = SigTo(value=lp_cutoff, time=0.05)
        self.filter = MoogLP(self.osc, freq=self.cutoff_sig, res=self.res_sig).out()
        self.is_finished = False # New flag to indicate full completion

    def stop(self):
        # Trigger the envelope release phase
        print(f"Triggering stop, envelope release time: {self.env.release}")
        self.env.stop()

        # Schedule cleanup after release time + small buffer
        # This CallAfter will trigger the cleanup of pyo objects and then mark the voice as finished.
        release_time = self.env.release
        self.cleanup_pattern = CallAfter(self._mark_as_finished, time=release_time + 0.05)
        self.cleanup_pattern.play()

    def _mark_as_finished(self):
        # Stop the pyo objects and free resources
        print("Cleaning up pyo objects for voice: stopping osc and filter")
        self.osc.stop()
        self.filter.stop()
        if hasattr(self, 'cleanup_pattern'):
            self.cleanup_pattern.stop()
        self.is_finished = True # Mark this voice as completely done

    def update_wavetable(self, new_tbl):
        self.osc.setTable(new_tbl)

    def update_filter(self, cutoff=None, resonance=None):
        if cutoff is not None:
            self.cutoff_sig.value = cutoff
        if resonance is not None:
            self.res_sig.value = resonance


class VoiceManager:
    def __init__(self, max_voices=8):
        self.voices = {}
        self.releasing_voices = [] # List to hold voices in their release phase
        self.max_voices = max_voices

        # Periodically clean up finished voices
        self.cleanup_pattern = Pattern(self._clean_finished_voices, time=0.1) # Check every 100ms
        self.cleanup_pattern.play()


    def _clean_finished_voices(self):
        # Filter out voices that have finished their release phase and are marked as 'is_finished'
        self.releasing_voices = [v for v in self.releasing_voices if not v.is_finished]
        # print(f"Active voices: {len(self.voices)}, Releasing voices: {len(self.releasing_voices)}")


    def note_on(self, note, voice):
        # Ensure cleanup runs before potentially stealing voices
        self._clean_finished_voices()

        if len(self.voices) + len(self.releasing_voices) >= self.max_voices:
            # Voice stealing:
            # 1. Prefer to steal from voices that are still playing (not yet in release)
            if self.voices:
                # Find the oldest playing voice (lowest note number, or simply the first in dict if ordered)
                # For simplicity, let's just pop the first one added if max_voices is hit
                oldest_note = next(iter(self.voices))
                oldest_voice = self.voices.pop(oldest_note)
                print(f"Stealing voice for note {oldest_note}")
                oldest_voice.stop() # Trigger its release
                self.releasing_voices.append(oldest_voice)
            # 2. If no playing voices, then force cleanup on an oldest releasing voice
            elif self.releasing_voices:
                oldest_releasing_voice = self.releasing_voices.pop(0) # Pop from the beginning (oldest)
                print(f"Forcing cleanup on releasing voice")
                oldest_releasing_voice._mark_as_finished() # Force its immediate cleanup

        self.voices[note] = voice

    def note_off(self, note):
        if note in self.voices:
            voice_to_release = self.voices.pop(note) # Remove from active voices
            voice_to_release.stop() # Trigger its release phase
            self.releasing_voices.append(voice_to_release) # Add to the list of releasing voices

    def update_all_filters(self, cutoff, resonance):
        for voice in self.voices.values():
            voice.update_filter(cutoff=cutoff, resonance=resonance)
        for voice in self.releasing_voices:
            voice.update_filter(cutoff=cutoff, resonance=resonance)

    def update_all_tables(self, new_tbl):
        for voice in self.voices.values():
            voice.update_wavetable(new_tbl)
        for voice in self.releasing_voices:
            voice.update_wavetable(new_tbl)

# (Rest of the code remains the same)
# =======================
# Main Synth Class
# =======================

class WavetableSynth:
    def __init__(self, server, wavetables):
        self.server = server
        self.wavetables = wavetables
        self.base_oscillator = "sin"
        self.mother_wavelet = "haar"
        self.table_bank = wavetables[self.base_oscillator][self.mother_wavelet]

        # Instead of Pyo's Midictl objects we maintain our own control variables.
        # These will be updated via Mido.
        self.one_hot_val = 0         # For CC 19 (wavetable selection, 0-3)
        self.base_period_val = 0     # For CC 20 (wavetable bank selection, up to 98)

        # ADSR values (from MIDI controls)
        self.attack_val = 0.01   # CC 73
        self.decay_val = 0.2     # CC 75
        self.sustain_val = 0.7   # CC 30
        self.release_val = 0.5   # CC 72

        # Low-pass filter cutoff (CC 74)
        self.lp_cutoff_val = 5000

        # Create smooth control signals using SigTo.
        self.attack_sig = SigTo(value=self.attack_val, time=0.05)
        self.decay_sig = SigTo(value=self.decay_val, time=0.05)
        self.sustain_sig = SigTo(value=self.sustain_val, time=0.05)
        self.release_sig = SigTo(value=self.release_val, time=0.05)
        self.lp_cutoff_sig = SigTo(value=self.lp_cutoff_val, time=0.05)

        # Instead of Pyo's Notein we'll use our own Sig objects for pitch and velocity.
        # They are updated manually via the MIDI listener.
        self.pitch = Sig(value=0)
        self.velocity = Sig(value=0)

        # Synth state for running oscillator, envelope, and filter.
        self.current_osc = None
        self.current_env = None
        self.current_lp_filter = None

        # Patterns to continuously update control signals.
        self.adsr_pattern = Pattern(self.update_adsr_params, time=0.05)
        self.adsr_pattern.play()
        self.lp_cutoff_pattern = Pattern(self.update_lp_cutoff, time=0.05)
        self.lp_cutoff_pattern.play()

        self.voice_manager = VoiceManager()

        # Resonance control (CC 71)
        self.res_val = 0.5
        # The SigTo objects should be updated when the CC value changes,
        # and then their *output* values are what's used by the voice manager
        self.res_sig = SigTo(value=self.res_val, time=0.05) # Keep this

        self.filter_params_pattern = Pattern(self.update_filter_params_in_voices, time=0.05)
        self.filter_params_pattern.play()

        # Octave shift (CC 14)
        self.octave_shift = 0  # -2 to +2

        # Continuous updates
        self.lp_cutoff_pattern = Pattern(self.update_filter_params, time=0.05)
        self.lp_cutoff_pattern.play()

        tbl = wavetables["sin"]["haar"][0][0]  # Just to start with something
        self.viewed_table = DataTable(size=tbl.getSize())
        self.viewed_table.replace(tbl.getTable())  # Fill with actual data
        self.viewed_table.view()


    def update_filter_params(self):
        self.lp_cutoff_sig.value = self.lp_cutoff_val
        self.res_sig.value = self.res_val
        self.voice_manager.update_all_filters(self.lp_cutoff_val, self.res_val)

    def update_filter_params_in_voices(self):
        # We pass the *raw* values here. The Voice's SigTo objects will handle smoothing.
        self.voice_manager.update_all_filters(self.lp_cutoff_val, self.res_val)

    # Renamed from update_lp_cutoff to reflect it's not just cutoff
    def update_adsr_params(self):
        # These are now updated directly in update_midi_cc
        pass # No need to do anything here if SigTo values are updated in update_midi_cc

    def update_lp_cutoff(self):
        self.lp_cutoff_sig.value = self.lp_cutoff_val
        if self.current_lp_filter is not None:
            self.current_lp_filter.freq = self.lp_cutoff_sig.value

    # def update_resonance(self, new_value):
    #     # Assume new_value is between 0.0 and 1.0
    #     self.res_sig.value = new_value

    def play_note(self, note, velocity):
        freq_val = MToF(Sig(note + self.octave_shift * 12))

        oh_idx = safe_index(self.one_hot_val, 0, len(self.table_bank) - 1)
        bp_idx = safe_index(self.base_period_val, 0, len(self.table_bank[0]) - 1)
        tbl = self.table_bank[oh_idx][bp_idx]

        adsr = (self.attack_val, self.decay_val, self.sustain_val, self.release_val)
        voice = Voice(freq=freq_val, velocity=velocity, tbl=tbl, adsr_params=adsr,
                      lp_cutoff=self.lp_cutoff_val, resonance=self.res_val)

        self.voice_manager.note_on(note, voice)


    def note_on_event(self, note, velocity):
        self.play_note(note, velocity)

    def note_off_event(self, note):
        self.voice_manager.note_off(note)

    def update_midi_cc(self, control, value):
        if control == 19:
            old_one_hot_val = self.one_hot_val
            self.one_hot_val = int(round((value / 127) * 3))
            if self.one_hot_val != old_one_hot_val:
                self._update_wavetable_all_voices()
        elif control == 20:
            old_base_period_val = self.base_period_val
            self.base_period_val = int(round((value / 127) * (len(self.table_bank[0]) - 1)))
            if self.base_period_val != old_base_period_val:
                self._update_wavetable_all_voices()
        elif control == 14:  # Octave shift
            self.octave_shift = int(value) - 2
        elif control == 71:  # Resonance
            self.res_val = value / 127 # Update the raw value
            self.res_sig.value = self.res_val # Update the SigTo directly here
        elif control == 73:  # Attack
            self.attack_val = (value / 127) * (2 - 0.001) + 0.001
            self.attack_sig.value = self.attack_val # Update SigTo
        elif control == 75:  # Decay
            self.decay_val = (value / 127) * (2 - 0.001) + 0.001
            self.decay_sig.value = self.decay_val # Update SigTo
        elif control == 30:  # Sustain
            self.sustain_val = value / 127
            self.sustain_sig.value = self.sustain_val # Update SigTo
        elif control == 72:  # Release
            self.release_val = max((value / 127) * (3 - 0.001) + 0.001, 0.02)
            self.release_sig.value = self.release_val # Update SigTo
        elif control == 74:  # Low-pass cutoff
            self.lp_cutoff_val = (value / 127) * (12000 - 20) + 20
            self.lp_cutoff_sig.value = self.lp_cutoff_val # Update SigTo
        print(f"Updated CC {control} with raw value {value}.")

    def _update_wavetable_all_voices(self):
        oh_idx = safe_index(self.one_hot_val, 0, len(self.table_bank) - 1)
        bp_idx = safe_index(self.base_period_val, 0, len(self.table_bank[0]) - 1)
        # This is where the *new* table is selected based on updated CC values
        tbl = self.table_bank[oh_idx][bp_idx]
        self.voice_manager.update_all_tables(tbl)  # This sends the new table to all active Voice objects

        self.viewed_table.replace(tbl.getTable())


# =======================
# Mido MIDI Listener Thread
# =======================

def midi_listener(synth, midi_port):
    """Continuously poll Mido for messages and forward them to the synth."""
    while True:
        for msg in midi_port.iter_pending():
            if msg.type == 'note_on' and msg.velocity > 0:
                synth.note_on_event(msg.note, msg.velocity)
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                synth.note_off_event(msg.note)
            elif msg.type == 'control_change':
                synth.update_midi_cc(msg.control, msg.value)
        time.sleep(0.01)


# =======================
# Main Entry Point
# =======================

def main():
    # Boot and start Pyo server.
    server = Server().boot()
    server.setMidiInputDevice(MIDI_DEVICE_INDEX)
    server.start()

    wavetables = load_wavetables(WAVETABLE_DIR)
    print("[Omnissiah] Wavetables loaded with due respect.")

    synth = WavetableSynth(server, wavetables)
    print("[Omnissiah] Synth ready to receive the divine MIDI signals.")

    # Open Mido's MIDI port.
    port_names = mido.get_input_names()
    midi_port = mido.open_input(port_names[MIDI_DEVICE_INDEX])
    print(f"[Omnissiah] Listening on MIDI port: {midi_port.name}")

    # Start the MIDI listener thread.
    t = threading.Thread(target=midi_listener, args=(synth, midi_port), daemon=True)
    t.start()

    # Set recording options
    # filename: The name of the output WAV file.
    # fileformat: The file format (0=WAV, 1=AIFF, 2=FLAC).
    # sampletype: The sample type (0=int16, 1=int24, 2=int32, 3=float32).
    # channels: Number of channels to record. Default is server's nchnls.
    # quality: Quality of the conversion (relevant for lossy formats like FLAC).
    # dur: Duration of the recording in seconds. If 0, it records until stopped.
    server.recordOptions(filename=os.path.join(output_dir, "0ne_h0t_s1nth_BASS.wav"),
                    fileformat=0,
                    sampletype=1)  # int24 is a good balance for quality and file size

    # Start the Pyo GUI (blocking call).
    server.gui(locals())

output_dir = "./recordings"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)





if __name__ == "__main__":
    main()