from pyo import *
import mido
import threading

# Boot the server
s = Server().boot()
s.start()

# Basic Triangle oscillator (for variety)
freq = Sig(440)  # Default to A4
t = SawTable(order=12).normalize()
osc = Osc(table=t, freq=freq, mul=.2)

# Create the ADSR envelope
env = Adsr(attack=0.1, decay=0.3, sustain=0.7, release=0.5, dur=1)

# Connect the envelope to the oscillator amplitude (mul)
osc.mul = env

# Add a low-pass filter
cutoff = Sig(1000)  # Default cutoff freq
filt = ButLP(osc, freq=cutoff)

# Output to speakers
filt.out()

# ---- MIDI Setup ----
midi_input_name = None
for name in mido.get_input_names():
    if 'LoopBe Internal MIDI' in name:
        midi_input_name = name
        break

if not midi_input_name:
    raise RuntimeError("Couldn't find S-1 MIDI input port.")

midi_in = mido.open_input(midi_input_name)

# MIDI handling in a thread
def midi_listener():
    for msg in midi_in:
        if msg.type == 'note_on' and msg.velocity > 0:
            # When a note_on message with velocity > 0 is received
            print(f"Note ON: {msg.note}")
            # Calculate the frequency from the MIDI note
            freq.value = 440.0 * (2 ** ((msg.note - 69) / 12.0))
            # Trigger the envelope (starts attack, then decay, sustain)
            env.play()

        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            # Handle note_off or note_on with velocity 0 (note-off equivalent)
            print(f"Note OFF: {msg.note}")
            # Stop the envelope by releasing it
            env.stop()  # Use .stop() instead of .release()

        elif msg.type == 'control_change':
            if msg.control == 49:  # Attack (CC49)
                attack_value = msg.value / 127.0 if msg.value != 0 else 0.01  # Prevent division by zero
                env.attack = attack_value
                print(f"Attack set to {env.attack}")

            elif msg.control == 75:  # Decay (CC75)
                decay_value = msg.value / 127.0 if msg.value != 0 else 0.01  # Prevent division by zero
                env.decay = decay_value
                print(f"Decay set to {env.decay}")

            elif msg.control == 30:  # Sustain (CC30)
                sustain_value = msg.value / 127.0 if msg.value != 0 else 0.01  # Prevent division by zero
                env.sustain = sustain_value
                print(f"Sustain set to {env.sustain}")

            elif msg.control == 72:  # Release (CC72)
                release_value = msg.value / 127.0 if msg.value != 0 else 0.01  # Prevent division by zero
                env.release = release_value
                print(f"Release set to {env.release}")

            elif msg.control == 74:  # Filter cutoff (CC74)
                norm_val = msg.value / 127.0 if msg.value != 0 else 0  # Prevent division by zero
                cutoff.value = 200 + norm_val * 8000  # Map to 200-8200 Hz
                print(f"Cutoff set to {cutoff.value}")

threading.Thread(target=midi_listener, daemon=True).start()

# Keep GUI alive
s.gui(locals())