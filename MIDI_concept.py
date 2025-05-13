import numpy as np
import sounddevice as sd
import mido

SAMPLE_RATE = 44100
DURATION = 0.5  # seconds


def note_to_freq_and_name(midi_note):
    # Define the standard 12-note scale
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Calculate the frequency of the note
    freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))

    # Calculate the note's position in the octave
    note_position = midi_note % 12

    # Determine the note name (C, D#, etc.)
    note_name = note_names[note_position]

    # Determine the octave (based on MIDI note number)
    octave = (midi_note // 12) - 1

    # Combine note name and octave
    full_note_name = f"{note_name}{octave}"

    return freq, full_note_name

def play_note(freq, duration=DURATION):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    sd.play(wave, samplerate=SAMPLE_RATE)
    sd.wait()

# List MIDI input ports
print("Available MIDI input ports:")
for name in mido.get_input_names():
    print(name)

# Choose your S1 port (adjust string match as needed)
port_name = next(name for name in mido.get_input_names() if 'S-1 0' in name)

print(f"Using MIDI input: {port_name}")
with mido.open_input(port_name) as inport:
    print("Listening for MIDI notes... Press Ctrl+C to quit.")
    for msg in inport:
        if msg.type == 'note_on' and msg.velocity > 0:
            freq, _ = note_to_freq_and_name(msg.note)
            print(f"Note {msg.note} → {freq:.2f} Hz")
            play_note(freq)


import mido

# List all available input ports
print("Available MIDI input ports:")
for port in mido.get_input_names():
    print(port)