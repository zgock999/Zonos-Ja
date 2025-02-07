# Zonos README (W.I.P)

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
