import mido
from pythonosc import udp_client
import time

# --- OSC Setup ---
# The IP address and port that the Pyo synthesizer will be listening on.
# Use '127.0.0.1' for localhost.
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 5005)

# --- mido Setup ---
midi_port_name = None
for name in mido.get_input_names():
    if "nanoPAD2" in name:
        midi_port_name = name
        break

if not midi_port_name:
    print("NanoPad 2 not found.")
    exit()

print(f"MIDI to OSC bridge listening on {midi_port_name}. Sending to localhost:5005")

def translate_midi_to_osc(msg):
    if msg.type == 'note_on':
        if msg.velocity > 0:
            osc_client.send_message("/note_on", [msg.note, msg.velocity])
            print(f"Sent /note_on {msg.note} {msg.velocity}")
        else:
            osc_client.send_message("/note_off", msg.note)
            print(f"Sent /note_off {msg.note}")
    elif msg.type == 'note_off':
        osc_client.send_message("/note_off", msg.note)
        print(f"Sent /note_off {msg.note}")
    elif msg.type == 'control_change':
        osc_client.send_message(f"/cc/{msg.control}", msg.value)
        print(f"Sent /cc/{msg.control} {msg.value}")

try:
    with mido.open_input(midi_port_name) as inport:
        for msg in inport:
            translate_midi_to_osc(msg)
except KeyboardInterrupt:
    print("\nBridge stopped.")
    pass