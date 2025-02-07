import torch
import torchaudio
from src.model import Zonos

model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
model.bfloat16().requires_grad_(False).eval()

torch.manual_seed(420)

conditioning = model.prepare_conditioning({
    "espeak": (["It would be nice to have time for testing, indeed."], ["en-us"]),
    "language_id": torch.tensor([24], device="cuda").view(1, 1, 1),  # 24 corresponds to "en-us"
})

codes = model.generate(conditioning)

wav = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wav[0], model.autoencoder.sampling_rate)
