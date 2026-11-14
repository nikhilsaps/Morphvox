"""Print audio devices with their indices, for choosing in/out in the monitor."""
import sounddevice as sd

print("INPUT devices (pick your real mic):")
for i, d in enumerate(sd.query_devices()):
    if d["max_input_channels"] > 0:
        print(f"  [{i:2}] {d['name']}")
print("\nOUTPUT devices (pick headphones for monitor; CABLE Input for VB-Cable):")
for i, d in enumerate(sd.query_devices()):
    if d["max_output_channels"] > 0:
        print(f"  [{i:2}] {d['name']}")
