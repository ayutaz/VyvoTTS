# VyvoTTS Windows環境構築ガイド

このドキュメントでは、Windows環境でVyvoTTSの推論を実行するための環境構築手順を説明します。

## 動作確認済み環境

| 項目 | バージョン |
|------|-----------|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (16GB VRAM) |
| CUDA Toolkit | 12.1 |
| Python | 3.12 |
| uv | 0.9.2 |

## 前提条件

- NVIDIA GPU（Compute Capability 8.0以上：RTX 30/40シリーズ、A100、H100など）
- CUDA Toolkit 12.x がインストール済み
- [uv](https://docs.astral.sh/uv/) がインストール済み

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/Vyvo-Labs/VyvoTTS.git
cd VyvoTTS
```

### 2. プロジェクトの初期化

```bash
uv init
```

### 3. Python バージョンの指定

```bash
uv python pin 3.12
```

> **注意**: `pyproject.toml`の`requires-python`が`>=3.13`になっている場合は、`>=3.12`に変更してください。

### 4. pyproject.toml の設定

`pyproject.toml`を以下のように設定します：

```toml
[project]
name = "vyvotts"
version = "0.1.0"
description = "LLM-Based Text-to-Speech Framework"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "accelerate>=1.12.0",
    "flash-attn>=2.7.4",
    "kernels>=0.11.5",
    "pyyaml>=6.0.3",
    "snac>=1.2.1",
    "soundfile>=0.13.1",
    "torch>=2.5.1",
    "torchaudio>=2.5.1",
    "torchvision>=0.20.1",
    "transformers>=4.57.3",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu124" }
torchvision = { index = "pytorch-cu124" }
torchaudio = { index = "pytorch-cu124" }
flash-attn = { url = "https://huggingface.co/lldacing/flash-attention-windows-wheel/resolve/main/flash_attn-2.7.4%2Bcu124torch2.6.0cxx11abiFALSE-cp312-cp312-win_amd64.whl" }
```

### 5. 依存パッケージのインストール

```bash
uv sync
```

これにより以下のパッケージがインストールされます：

| パッケージ | バージョン | 用途 |
|-----------|-----------|------|
| torch | 2.6.0+cu124 | 深層学習フレームワーク |
| transformers | 4.57.3 | LLMモデル読み込み |
| flash-attn | 2.7.4+cu124 | Flash Attention 2（高速化） |
| snac | 1.2.1 | 音声コーデック（SNAC） |
| accelerate | 1.12.0 | モデル読み込み高速化 |
| soundfile | 0.13.1 | WAVファイル保存 |
| pyyaml | 6.0.3 | 設定ファイル読み込み |

### 6. 動作確認

```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

期待される出力：
```
PyTorch: 2.6.0+cu124
CUDA: True
```

## 推論の実行

### 推論スクリプトの作成

`scripts/inference_test.py`を作成：

```python
import torch
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import time
import soundfile as sf


def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    print("Loading configuration...")
    config = load_config("vyvotts/configs/inference/lfm2.yaml")

    # Token constants from config
    START_OF_HUMAN = config['START_OF_HUMAN']
    END_OF_TEXT = config['END_OF_TEXT']
    END_OF_HUMAN = config['END_OF_HUMAN']
    START_OF_SPEECH = config['START_OF_SPEECH']
    END_OF_SPEECH = config['END_OF_SPEECH']
    AUDIO_TOKENS_START = config['AUDIO_TOKENS_START']

    device = "cuda"
    model_name = "Vyvo/VyvoTTS-LFM2-Neuvillette"

    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to(device)

    print("Loading LLM model with Flash Attention 2...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Input text
    text = "Hello, this is a test of the VyvoTTS speech synthesis system."
    print(f"Generating speech for: {text}")

    # Preprocess
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1).to(device)
    attention_mask = torch.ones_like(modified_input_ids)

    # Generate
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=modified_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=END_OF_SPEECH,
        )

    torch.cuda.synchronize()
    generation_time = time.time() - start_time

    # Parse audio tokens
    token_indices = (generated_ids == START_OF_SPEECH).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        last_idx = token_indices[1][-1].item()
        cropped = generated_ids[:, last_idx+1:]
    else:
        cropped = generated_ids

    row = cropped[0]
    row = row[row != END_OF_SPEECH]
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed = row[:new_length]
    code_list = [t.item() - AUDIO_TOKENS_START for t in trimmed]

    # Redistribute codes to SNAC layers
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))

    codes = [
        torch.tensor(layer_1).unsqueeze(0).to(device),
        torch.tensor(layer_2).unsqueeze(0).to(device),
        torch.tensor(layer_3).unsqueeze(0).to(device)
    ]

    # Decode audio
    audio = snac_model.decode(codes)
    audio_numpy = audio.detach().squeeze().cpu().numpy()

    # Save
    output_path = "test_output.wav"
    sf.write(output_path, audio_numpy, 24000)

    print(f"Audio shape: {audio.shape}")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
```

### 実行

```bash
uv run python scripts/inference_test.py
```

初回実行時はモデルのダウンロード（約1.4GB）が行われます。

### 期待される出力

```
Loading configuration...
Loading SNAC model...
Loading LLM model with Flash Attention 2...
Generating speech for: Hello, this is a test of the VyvoTTS speech synthesis system.
Audio shape: torch.Size([1, 1, 98304])
Generation time: 9.44s
Saved to: test_output.wav
```

## Windows固有の注意事項

### Flash Attention について

- **Flash Attention 3**（`kernels-community/flash-attn3`）はWindows未対応
- 代わりに**Flash Attention 2**を使用
- Windows用のプリビルドwheelは[lldacing/flash-attention-windows-wheel](https://huggingface.co/lldacing/flash-attention-windows-wheel)から取得

### CUDA バージョンについて

- Flash Attention 2のWindows wheelがCUDA 12.4向けのため、PyTorchもCUDA 12.4版を使用
- CUDA Toolkit 12.1がインストールされていても、PyTorch CUDA 12.4は動作可能（後方互換性）

### vLLM について

- vLLMはLinux専用のため、Windows環境では使用不可
- Windows環境ではTransformers + Flash Attention 2を使用

## トラブルシューティング

### `ModuleNotFoundError: No module named 'vyvotts'`

`pyproject.toml`に`[build-system]`セクションを追加し、`uv sync`を実行してください。

### Flash Attention のインストールエラー

`pyproject.toml`の`[tool.uv.sources]`セクションでflash-attnのURLが正しく設定されていることを確認してください。

### CUDA関連のエラー

```bash
nvidia-smi
```
でGPUが認識されていることを確認してください。

## 参考リンク

- [VyvoTTS GitHub](https://github.com/Vyvo-Labs/VyvoTTS)
- [Flash Attention Windows Wheels](https://huggingface.co/lldacing/flash-attention-windows-wheel)
- [uv Documentation](https://docs.astral.sh/uv/)
