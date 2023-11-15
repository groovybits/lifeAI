import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
#import uuid

model = MusicGen.get_pretrained('facebook/musicgen-stereo-small')
model.set_generation_params(duration=30)
wav = model.generate_unconditional(1)
descriptions = ['naruto anime intro with excitement and action japanese rock']
wav = model.generate(descriptions)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    #id=uuid.uuid4().replace('-','')
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
