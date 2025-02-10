# Zonos-v0.1

<div align="center">
<img src="content/ZonosHeader.png" 
     alt="Alt text" 
     style="width: 500px;
            height: auto;
            object-position: center top;">
</div>

Zonos-v0.1 is a leading open-weight text-to-speech model, delivering expressiveness and quality on par with—or even surpassing—top TTS providers.

It enables highly naturalistic speech generation from text prompts when given a speaker embedding or audio prefix. With just 5 to 30 seconds of speech, Zonos can achieve high-fidelity voice cloning. It also allows conditioning based on speaking rate, pitch variation, audio quality, and emotions such as sadness, fear, anger, happiness, and joy. The model outputs speech natively at 44kHz.

Trained on approximately 200,000 hours of primarily English speech data, Zonos follows a straightforward architecture: text normalization and phonemization via eSpeak, followed by DAC token prediction through a transformer or hybrid backbone. An architecture overview can be seen below.

<div align="center">
<img src="content/ArchitectureDiagram.png" 
     alt="Alt text" 
     style="width: 1000px;
            height: auto;
            object-position: center top;">
</div>

Read more about our models [here](https://www.zyphra.com/post/beta-release-of-zonos-v0-1).

## Features
* Zero-shot TTS with voice cloning: Input desired text and a 10-30s speaker sample to generate high quality TTS output
* Audio prefix inputs: Add text plus an audio prefix for even richer speaker matching. Audio prefixes can be used to elicit behaviours such as whispering which are challenging to obtain from pure voice cloning
* Multilingual support: Zonos-v0.1 supports English, Japanese, Chinese, French, and German
* Audio quality and emotion control: Zonos offers fine-grained control of many aspects of the generated audio. These include speaking rate, pitch, maximum frequency, audio quality, and various emotions such as happiness, anger, sadness, and fear.
* Fast: our model runs with a real-time factor of ~2x on an RTX 4090
* WebUI gradio interface: Zonos comes packaged with an easy to use gradio interface to generate speech 
* Simple installation and deployment: Zonos can be installed and deployed simply using the docker file packaged with our repository.


## Docker Installation

```bash
git clone git@github.com:Zyphra/Zonos.git
cd Zonos

# For gradio
docker compose up

# Or for development you can do
docker build -t Zonos .
docker run -it --gpus=all --net=host -v /path/to/Zonos:/Zonos -t Zonos
cd /Zonos
python3 sample.py # this will generate a sample.wav in /Zonos
```

## DIY Installation
### eSpeak

```bash
apt install espeak-ng
```

### Python dependencies

Make sure you have a recent version of [uv](https://docs.astral.sh/uv/#installation), then run the following commands in sequence:

```bash
uv venv
uv sync --no-group main
uv sync
```

## Usage example

```bash
Python3 sample.py
```
This will produce `sample.wav` in the `Zonos` directory.

## Getting started with Zonos in python
Once you have Zonos installed try generating audio programmatically in python
```python3
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

# Use the hybrid with "Zyphra/Zonos-v0.1-hybrid"
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
```
