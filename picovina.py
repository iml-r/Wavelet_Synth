from pyo import *

# Boot server with explicit MIDI device
s = Server()
s.setMidiInputDevice(2)
s.boot()
s.start()

# Basic MIDI input
note = Notein(poly=8)
pitch = note['pitch']
velocity = note['velocity']
gate = note['trigon']

# Convert MIDI note to frequency
freq = MToF(pitch)
amp = velocity / 127.0

# Simple envelope and oscillator
env = Adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.2, dur=2, mul=amp)
sine = Sine(freq=freq, mul=env).out()

# Trigger envelope on key press
trig = TrigFunc(gate, function=env.play)

s.gui(locals())