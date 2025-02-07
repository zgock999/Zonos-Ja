# Zonos README

Zonos-v0.1-transformer is a leading open-weight text-to-speech transformer model. In our testing we have found it performs comparably or better in expressiveness and quality compared to leading TTS providers.

Zonos enables highly expressive and naturalistic speech generation from text prompts given a speaker embedding or audio prefix. Zonos is capable of high fidelity voice cloning given clips of between 5 and 30s of speech. Zonos also can be conditioned based on speaking rate, pitch standard deviation, audio quality, and emotions such as sadness, fear, anger, happiness, and joy. Zonos outputs speech natively at 44Khz.

Zonos was trained on approximately 200k hours of primarily English speech data.

Zonos follows a simple architecture comprising text normalisation and phonemization by espeak, followed by DAC token prediction by a transformer backbone.

Read more about our models [here](https://www.zyphra.com/post/beta-release-of-zonos-v0-1).

## Installation

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
uv run sample.py
```
This will produce `sample.wav` in the `Zonos` directory.
