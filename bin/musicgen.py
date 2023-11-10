import torch
from transformers import pipeline

synthesiser = pipeline("text-to-audio",
    "facebook/musicgen-stereo-medium",
    device="mps",
    torch_dtype=torch.float16)

music = synthesiser("EDM song that reminds me of a banana!")

print(f"Music generated has {music['sampling_rate']} sampling rate saving to musicgen_out.wav...")

sf.write("musicgen_out.wav", music["audio"][0].T, music["sampling_rate"])

