import mido

def handle(msg):
    print("Received:", msg)

# List available ports and pick the right one (based on what you saw in MIDI-OX)
ports = mido.get_input_names()
print(f"Available ports: {ports}")
if not ports:
    raise RuntimeError("No MIDI input devices found!")

port_name = ports[0]  # Or choose the one matching your device, e.g., "S-1 0"

# Open the port and listen to MIDI messages
try:
    with mido.open_input(port_name, callback=handle):
        print(f"Listening on {port_name}... Press Ctrl+C to quit")
        import time
        while True:
            time.sleep(1)  # Keep the script alive for MIDI to flow
except Exception as e:
    print(f"Error opening port: {e}")