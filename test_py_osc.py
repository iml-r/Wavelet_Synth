import mido
from pythonosc import udp_client

# Set your actual input port name here (mido.get_input_names() will help)
midi_input_name = 'nanoPAD2 1'

# Setup OSC client to send to pyo on localhost:12345
osc_client = udp_client.SimpleUDPClient('127.0.0.1', 12345)

with mido.open_input(midi_input_name) as inport:
    print(f"Listening to {midi_input_name}...")
    for msg in inport:
        # Send note_on and note_off events as OSC
        if msg.type == 'note_on':
            osc_client.send_message('/note_on', [msg.note, msg.velocity])
        elif msg.type == 'note_off':
            osc_client.send_message('/note_off', [msg.note])
        # You can add CC or other messages here similarly