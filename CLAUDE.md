# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

VyvoTTSはLLM（大規模言語モデル）をベースにしたText-to-Speech（TTS）トレーニング・推論フレームワークです。LiquidAI LFM2やQwen3などのモデルを使用して、音声合成を行います。

## セットアップ

### Windows環境（推論のみ）

詳細は [docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md) を参照。

```bash
uv init
uv python pin 3.12
uv sync
```

### Linux環境（トレーニング含む）

```bash
uv venv --python 3.10
uv pip install -r requirements.txt
```

## 主要コマンド

### データセット準備（オーディオトークン化）
```python
from vyvotts.audio_tokenizer import process_dataset

process_dataset(
    original_dataset="HuggingFace/dataset-name",
    output_dataset="username/output-name",
    model_type="lfm2",  # または "qwen3"
    text_field="text"
)
```

### ファインチューニング
```bash
# 設定ファイル: vyvotts/configs/train/lfm2_ft.yaml
accelerate launch --config_file vyvotts/configs/train/accelerate_finetune.yaml vyvotts/train/finetune/train.py
```

### プリトレーニング（マルチGPU FSDP）
```bash
# 設定ファイル: vyvotts/configs/train/lfm2_pretrain.yaml
accelerate launch --config_file vyvotts/configs/train/accelerate_pretrain.yaml vyvotts/train/pretrain/train.py
```

### 推論

#### Windows環境（Flash Attention 2使用）
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Vyvo/VyvoTTS-LFM2-Neuvillette",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
```

#### Linux環境
```python
# Transformers（標準）
from vyvotts.inference.transformers_inference import VyvoTTSTransformersInference
engine = VyvoTTSTransformersInference(model_name="Vyvo/VyvoTTS-LFM2-Neuvillette")
audio, timing = engine.generate(text="Hello", output_path="output.wav")

# vLLM（高速・Linux専用）
from vyvotts.inference.vllm_inference import VyvoTTSInference
engine = VyvoTTSInference(model_name="Vyvo/VyvoTTS-LFM2-Neuvillette")
audio = engine.generate(text="Hello", voice="zoe", output_path="output.wav")

# Unsloth（省メモリ、4bit/8bit量子化）
from vyvotts.inference.unsloth_inference import VyvoTTSUnslothInference
engine = VyvoTTSUnslothInference(model_name="Vyvo/VyvoTTS-v2-Neuvillette", load_in_4bit=True)

# HQQ（高品質量子化）
from vyvotts.inference.transformers_hqq_inference import VyvoTTSHQQInference
engine = VyvoTTSHQQInference(model_name="Vyvo/VyvoTTS-LFM2-Neuvillette", nbits=4)
```

## アーキテクチャ

### オーディオエンコーディング
- **SNACコーデック**: `hubertsiuzdak/snac_24khz`を使用
- 3階層の階層的ベクトル量子化（1+2+4=7コード/フレーム）
- オーディオトークンは`AUDIO_TOKENS_START`からのオフセットで管理

### トークン構造（lfm2設定）
```
入力形式: [START_OF_HUMAN] テキスト [END_OF_TEXT] [END_OF_HUMAN] [START_OF_AI] [START_OF_SPEECH] 音声コード [END_OF_SPEECH] [END_OF_AI]
```

特殊トークンはYAML設定（`vyvotts/configs/inference/lfm2.yaml`等）で定義:
- `AUDIO_TOKENS_START`: 64410（音声トークンのベースオフセット）
- 各コードブックレイヤーは4096のオフセット間隔

### 設定ファイル
- **推論設定**: `vyvotts/configs/inference/{lfm2,qwen3,llama3}.yaml` - モデル固有のトークン定義
- **訓練設定**: `vyvotts/configs/train/{lfm2_ft,lfm2_pretrain,qwen3_ft}.yaml` - ハイパーパラメータ
- **accelerate設定**: `vyvotts/configs/train/accelerate_*.yaml` - 分散訓練設定

### プリトレーニングの特徴
- `GradualRatioDataset`: テキストQAと音声データの比率を訓練中に段階的に変更（例: 2:1 → 1:1）
- FSDP（Fully Sharded Data Parallel）による大規模分散訓練
- Liger Kernelによる最適化

### ボイスクローニング
`vyvotts/voice_clone.py`でリファレンス音声を使った音声合成が可能:
```python
from vyvotts.voice_clone import VyvoTTSVoiceClone
cloner = VyvoTTSVoiceClone()
audio = cloner.clone_voice("reference.wav", "transcript", ["target text"])
```

## サポートモデル

- **LFM2**: LiquidAI/LFM2-350M（推奨）
- **Qwen3**: Qwen/Qwen3-0.6B
- ファインチューン済み: `Vyvo/VyvoTTS-LFM2-Neuvillette`, `Vyvo/VyvoTTS-v2-Neuvillette`

## 低VRAM環境向け

6GB+ VRAMの場合は`notebook/vyvotts-lfm2-train.ipynb`でUnsloth FP8/FP4トレーニングを使用。
