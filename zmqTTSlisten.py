#!/usr/bin/env python

## Life AI Text to Speech listener ZMQ client
#
# Chris Kennedy 2023 (C) GPL
#
# Free to use for any use as in truly free software
# as Richard Stallman intended it to be.
#

import io
import zmq
import argparse
import textwrap
import soundfile as sf

def main(input_port, output_file=None, audio_format=None):
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.connect(f"tcp://localhost:{input_port}")

    while True:
        try:
            # Receive the segment number (header) first
            header_str = socket.recv_string()

            # Now, receive the binary audio data
            payload_bytes = socket.recv()

            # Print the header
            print(f"Header: {header_str}")

            # Check if we need to output to a file
            if output_file:
                if audio_format == "wav":
                    #audiobuf = io.BytesIO()
                    #sf.write(audiobuf, payload_bytes, 16000, format='WAV')
                    #audiobuf.seek(0)
                    with open(output_file, 'wb') as f:
                        f.write(payload_bytes)
                    print(f"Audio saved to {output_file} as WAV")
                else:
                    with open(output_file, 'wb') as f:
                        f.write(payload_bytes)
                    print(f"Payload written to {output_file}\n")
            else:
                # Convert the payload to its hex representation and display
                payload_hex = payload_bytes.hex()
                print(f"Payload (Hex): {textwrap.fill(payload_hex, width=80)}\n")


        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_port", type=int, required=True, help="Port for receiving audio numpy arrays")
    parser.add_argument("--output_file", type=str, help="Path to save the received audio")
    parser.add_argument("--audio_format", type=str, choices=["wav", "raw"], default="raw", help="Audio format to save as. Choices are 'wav' or 'raw'. Default is 'raw'.")
    args = parser.parse_args()

    main(args.input_port, args.output_file, args.audio_format)

