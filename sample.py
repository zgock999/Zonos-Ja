import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
model.bfloat16()

wav, sampling_rate = torchaudio.load("./exampleaudio.mp3")
spk_embedding = model.embed_spk_audio(wav, sampling_rate)

torch.manual_seed(421)

cond_dict = make_cond_dict(
    text="Hello, world!",
    speaker=spk_embedding.to(torch.bfloat16),
    language="en-us",
)
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
