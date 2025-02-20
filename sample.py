import torch
import torchaudio
import warnings
import os
import sys
import argparse
import random
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# コンパイル最適化の設定（シンプル化）
os.environ["TORCH_INDUCTOR_VERBOSE"] = "0"
os.environ["TORCH_COMPILE_DEBUG"] = "0"
torch._dynamo.config.suppress_errors = True

# venv環境の検出とキャッシュディレクトリの設定
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    # venv環境内の場合
    venv_path = sys.prefix
    cache_dir = os.path.join(venv_path, 'var', 'torch_compile')
else:
    # venv環境でない場合
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch_compile")

os.makedirs(cache_dir, exist_ok=True)
os.environ["TORCH_COMPILE_CACHE_DIR"] = cache_dir

# コマンドライン引数の解析
parser = argparse.ArgumentParser()
parser.add_argument('--fp16', action='store_true', help='Enable fp16 computation')
parser.add_argument('--seed', type=int, default=421, help='Random seed (default: 421)')
args = parser.parse_args()

# シード値の設定
if args.seed < 0:
    args.seed = random.randint(0, 2**32 - 1)
    print(f"Negative seed provided, using random seed: {args.seed}")
else:
    print(f"Using provided seed: {args.seed}")

torch.manual_seed(args.seed)

# 警告抑制とGPU設定は共通
warnings.filterwarnings("ignore", category=UserWarning, module="torch._inductor")
warnings.filterwarnings("ignore", message=".*Not enough SMs.*")

# GPUメモリとCUDA設定
torch.cuda.set_per_process_memory_fraction(0.95)
torch.backends.cudnn.benchmark = True

# fp16が指定された場合のみ追加設定を適用
if args.fp16:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_INDUCTOR_MAX_AUTOTUNE"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# モデルの読み込みとfp16変換（条件付き）
model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
if args.fp16:
    model = model.half()
    for param in model.parameters():
        param.data = param.data.half()
    for buf in model.buffers():
        buf.data = buf.data.half()

# モデルをコンパイル（シンプル化）
model = torch.compile(
    model,
    backend="inductor",
    mode="default",
    fullgraph=False
)

wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")

# speaker embeddingの処理
if args.fp16:
    with torch.amp.autocast('cuda', enabled=False, dtype=torch.float32, cache_enabled=False):
        speaker = model.make_speaker_embedding(wav.to(device), sampling_rate)
    speaker = speaker.to(dtype=torch.float16)
else:
    speaker = model.make_speaker_embedding(wav.to(device), sampling_rate)

cond_dict = make_cond_dict(text="こんにちわ！世界よ！", speaker=speaker, language="ja")
if args.fp16:
    for k, v in cond_dict.items():
        if torch.is_tensor(v) and not v.dtype.is_floating_point:
            continue
        if torch.is_tensor(v):
            cond_dict[k] = v.to(dtype=torch.float16)

conditioning = model.prepare_conditioning(cond_dict)

# 生成処理
if args.fp16:
    with torch.amp.autocast('cuda', dtype=torch.float16, cache_enabled=False):
        codes = model.generate(conditioning)
        wavs = model.autoencoder.decode(codes)
else:
    codes = model.generate(conditioning)
    wavs = model.autoencoder.decode(codes)

wavs = wavs.cpu().float()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
