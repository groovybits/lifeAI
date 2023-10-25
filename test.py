import torch
import torchaudio
from seamless_communication.models.inference import Translator

# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_large", vocoder_name_or_card="vocoder_36langs", device=torch.device("cuda:0"))

# We got the languages above from the official paper
src_lang = "tgl" #tagalog
tgt_lang = "eng" #english
input_text = "Salamat sa MetaAI at naglabas sila SeamlessM4T model para gamitin ng mga tao!"
translated_text, wav, sr = translator.predict(input_text, "t2st", tgt_lang=tgt_lang, src_lang=src_lang)

#Let's print the translated text
print(translated_text)

# Save the translated audio generation.
torchaudio.save(
    "Tagalog-to-English.wav",
    wav[0].cpu(),
    sample_rate=sr,
)
