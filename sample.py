import torch
import torchaudio
from zonos.model import Zonos

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
model.bfloat16()

wav, sampling_rate = torchaudio.load("./britishmale3.mp3")
spk_embedding = model.embed_spk_audio(wav, sampling_rate)

torch.manual_seed(421)

conditioning = model.prepare_conditioning(
    {
        "espeak": (["It would be nice to have time for testing, indeed."], ["en-us"]),
        "language_id": torch.tensor([24], device="cuda").view(1, 1, 1),  # 24 corresponds to "en-us"
        "speaker": spk_embedding.to(torch.bfloat16),
    }
)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
