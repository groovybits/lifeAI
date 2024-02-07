#!/usr/bin/env python3

import argparse
import scipy
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from IPython.display import Audio

def main(tokens, music_values, output_file, model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)

    inputs = processor(
        text=music_values,
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs, max_new_tokens=tokens)
    print(f"Generated audio_values from {tokens} tokens")

    sampling_rate = model.config.audio_encoder.sampling_rate
    print(f"Sampling rate is {sampling_rate}")
    scipy.io.wavfile.write(output_file, rate=sampling_rate, data=audio_values[0, 0].numpy())
    print(f"Generated music saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate music with specified parameters.')
    parser.add_argument('--tokens', type=int, default=500, help='Number of tokens to generate.')
    parser.add_argument('--music_values', nargs='+', type=str, required=True, help='List of music values (prompts).')
    parser.add_argument('--output_file', type=str, default="musicgen_out.wav", help='Output file name for the generated music.')
    parser.add_argument('--model_name', type=str, default="facebook/musicgen-small", help='Model name to use for generation.')

    args = parser.parse_args()

    main(args.tokens, args.music_values, args.output_file, args.model_name)
