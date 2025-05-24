import mido

print("Available MIDI input ports:")
for name in mido.get_input_names():
    print(f" - {name}")

# Replace with your device name, or use the first available port:
port_name = mido.get_input_names()[1]  # or "Roland S-1" etc.

print(f"\nListening on {port_name} ... Press Ctrl+C to stop.")


with mido.open_input(port_name) as inport:
    for msg in inport:
        if msg.type in ('clock', 'start', 'stop', 'continue'):
            continue  # ignore MIDI clock messages
        print(msg)