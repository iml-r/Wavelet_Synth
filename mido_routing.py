# midi_to_osc.py
import mido
from pythonosc import udp_client

# Replace with your actual MIDI input port name (check with mido.get_input_names())
midi_input_name = "nanoPAD2 1"

# Create OSC client to send messages to pyo server (localhost:12345)
osc_client = udp_client.SimpleUDPClient("127.0.0.1", 12345)

print(f"Opening MIDI input: {midi_input_name}")
with mido.open_input(midi_input_name) as inport:
    print("Listening for MIDI... Press Ctrl+C to quit.")
    for msg in inport:
        # Filter and forward MIDI messages you care about

        if msg.type == 'note_on':
            osc_client.send_message("/note_on", [msg.note, msg.velocity])
            print(f"note_on: note={msg.note} vel={msg.velocity}")

        elif msg.type == 'note_off':
            osc_client.send_message("/note_off", [msg.note])
            print(f"note_off: note={msg.note}")

        elif msg.type == 'control_change':
            osc_client.send_message("/cc", [msg.control, msg.value])
            print(f"cc: control={msg.control} val={msg.value}")

        # Add more if needed...