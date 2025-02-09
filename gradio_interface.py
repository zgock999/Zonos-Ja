import torch
import torchaudio
import gradio as gr

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict

device = "cuda"

language_codes = [
    'af', 'am', 'an', 'ar', 'as', 'az', 'ba', 'bg', 'bn', 'bpy', 'bs', 'ca', 'cmn',
    'cs', 'cy', 'da', 'de', 'el', 'en-029', 'en-gb', 'en-gb-scotland', 'en-gb-x-gbclan',
    'en-gb-x-gbcwmd', 'en-gb-x-rp', 'en-us', 'eo', 'es', 'es-419', 'et', 'eu', 'fa',
    'fa-latn', 'fi', 'fr-be', 'fr-ch', 'fr-fr', 'ga', 'gd', 'gn', 'grc', 'gu', 'hak',
    'hi', 'hr', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'is', 'it', 'ja', 'jbo', 'ka',
    'kk', 'kl', 'kn', 'ko', 'kok', 'ku', 'ky', 'la', 'lfn', 'lt', 'lv', 'mi', 'mk',
    'ml', 'mr', 'ms', 'mt', 'my', 'nb', 'nci', 'ne', 'nl', 'om', 'or', 'pa', 'pap',
    'pl', 'pt', 'pt-br', 'py', 'quc', 'ro', 'ru', 'ru-lv', 'sd', 'shn', 'si', 'sk',
    'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tn', 'tr', 'tt', 'ur', 'uz', 'vi',
    'vi-vn-x-central', 'vi-vn-x-south', 'yue'
]

# Global variables to track the currently loaded model.
CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

def load_model_if_needed(model_choice: str) -> Zonos:
    """
    Loads the requested model if it is not already loaded.
    Frees VRAM for the previous model if a different model is requested.
    """
    global CURRENT_MODEL_TYPE, CURRENT_MODEL

    if CURRENT_MODEL_TYPE != model_choice:
        # Free the previous model if one was loaded.
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()

        print(f"Loading {model_choice} model...")
        if model_choice == "Transformer":
            CURRENT_MODEL = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
        else:
            CURRENT_MODEL = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
        CURRENT_MODEL.to(device)
        CURRENT_MODEL.bfloat16()  # Remove or switch to float16() if needed.
        CURRENT_MODEL.eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    else:
        print(f"{model_choice} model is already loaded.")

    return CURRENT_MODEL

def generate_audio(
    model_choice,
    text,
    language,
    speaker_audio,
    prefix_audio,
    fmax,
    pitch_std,
    speaking_rate,
    ctc_loss,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    min_p,
    seed,
):
    selected_model = load_model_if_needed(model_choice)

    speaker_embedding = None
    if speaker_audio is not None:
        wav, sr = torchaudio.load(speaker_audio)
        speaker_embedding = selected_model.embed_spk_audio(wav, sr)
        speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)

    audio_prefix_codes = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = torchaudio.functional.resample(wav_prefix, sr_prefix, selected_model.autoencoder.sampling_rate)
        with torch.autocast(device, dtype=torch.float32):
            audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0).to(device))

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    ctc_loss = float(ctc_loss)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    min_p = float(min_p)
    seed = int(seed)
    max_new_tokens = 86 * 30
    batch_size = 1

    torch.manual_seed(seed)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=speaker_embedding,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        ctc_loss=ctc_loss,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised_bool,
        device=device,
    )

    conditioning = selected_model.prepare_conditioning(cond_dict)
    sampling_params = dict(min_p=min_p)
    codes = selected_model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=batch_size,
        sampling_params=sampling_params,
    )

    wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
    sr_out = selected_model.autoencoder.sampling_rate

    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]

    return sr_out, wav_out.squeeze().numpy()

def build_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=["Hybrid", "Transformer"],
                    value="Transformer",
                    label="Zonos Model Type",
                    info="Select the model variant to use."
                )
                text = gr.Textbox(
                    label="Text to Synthesize",
                    value="Zonos follows a simple architecture comprising text normalization and phonemization by eSpeak.",
                    lines=4,
                )
                language = gr.Dropdown(
                    choices=language_codes,
                    value="en-us",
                    label="Language Code",
                    info="Select a language code.",
                )
            # New audio input for prefix codes
            prefix_audio = gr.Audio(
                label="Optional Prefix Audio (to continue from)",
                value="./silence_100ms.wav",
                type="filepath",
            )
            # Speaker audio (for voice cloning) 
            speaker_audio = gr.Audio(
                label="Optional Speaker Audio (for cloning)",
                type="filepath",
            )

        with gr.Column():
            gr.Markdown("## Conditioning Parameters")
            with gr.Row():
                fmax_slider = gr.Slider(1000, 48000, value=22050, step=1, label="fmax (Hz)")
                pitch_std_slider = gr.Slider(0.0, 100.0, value=20.0, step=0.1, label="pitch_std")
                speaking_rate_slider = gr.Slider(1.0, 30.0, value=15.0, step=0.1, label="speaking_rate")
            with gr.Row():
                ctc_loss_slider = gr.Slider(0.0, 10.0, value=0.0, step=0.01, label="ctc_loss")
                dnsmos_slider = gr.Slider(0.0, 5.0, value=4.0, step=0.1, label="dnsmos_ovrl")
                speaker_noised_checkbox = gr.Checkbox(label="Denoise speaker embedding?", value=False)

        with gr.Column():
            gr.Markdown("## Generation Parameters")
            with gr.Row():
                cfg_scale_slider = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="cfg_scale")
                min_p_slider = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="min_p")
                seed_number = gr.Number(label="Seed (for reproducibility)", value=420, precision=0)

        generate_button = gr.Button("Generate Audio")
        output_audio = gr.Audio(label="Generated Audio", type="numpy")

        generate_button.click(
            fn=generate_audio,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                prefix_audio,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                ctc_loss_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                min_p_slider,
                seed_number,
            ],
            outputs=[output_audio],
        )

    return demo

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)