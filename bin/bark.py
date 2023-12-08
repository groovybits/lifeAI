#!/usr/bin/env python

from transformers import AutoProcessor, BarkModel
import torch
import scipy

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "mps"

processor = AutoProcessor.from_pretrained("suno/bark-small")
model = BarkModel.from_pretrained("suno/bark-small").to(device)
sample_rate = model.generation_config.sample_rate

voice_preset = "v2/en_speaker_6"
inputs = processor(
    "[fart] Welcome to the Groovy Life AI wacky funtime hour! [clapping] Today we have a great show in store for you!!! [laughing] [clapping]", 
    voice_preset=voice_preset).to(device)
#encoder_outputs = text_encoder(**inputs)
audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()

scipy.io.wavfile.write("bark_out2.wav", rate=sample_rate, data=audio_array)

