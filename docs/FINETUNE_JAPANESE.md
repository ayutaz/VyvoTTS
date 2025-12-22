# VyvoTTS 日本語ファインチューニングガイド

VyvoTTSを日本語に対応させるためのファインチューニング手順をまとめます。

## 概要

VyvoTTSは現在英語のみ対応ですが、日本語音声データセットでファインチューニングすることで日本語対応が可能です。

### 前提条件

| 項目 | 要件 |
|------|------|
| トークナイザー | LFM2/Qwen3（日本語対応済み） |
| SNACコーデック | 言語非依存（日本語音声エンコード可能） |
| ベースモデル | VyvoTTS-LFM2-350M-PT（推奨） |

---

## 1. 必要なリソース

### GPU要件

| GPU | VRAM | 推定学習時間（10時間データ） |
|-----|------|---------------------------|
| **RTX 4090** | 24GB | 2〜4時間 |
| **T4 x 4** | 16GB x 4 | 1.5〜3時間 |
| RTX 4070 Ti SUPER | 16GB | 4〜8時間 |
| T4（単体） | 16GB | 6〜12時間 |

### データセット要件

| レベル | データ量 | 品質 |
|--------|----------|------|
| 最小 | 50サンプル | テスト用 |
| 推奨 | 300サンプル/話者 | 良好 |
| 実用 | 3〜10時間 | 高品質 |
| 本格 | 30時間以上 | 多話者対応 |

### 利用可能な日本語データセット

| データセット | 時間 | 話者数 | URL |
|--------------|------|--------|-----|
| **JSUT** | ~10時間 | 1人（女性） | [Shinnolab](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) |
| **JVS** | ~30時間 | 100人 | [Shinnolab](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) |
| **JVNV** | - | 多数 | [Shinnolab](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvnv_corpus) |

---

## 2. 環境構築

### 2.1 依存関係のインストール

```bash
# uvを使用する場合
uv sync

# pipを使用する場合
pip install -r requirements.txt
```

### 2.2 追加パッケージ

```bash
# 学習に必要なパッケージ
uv add accelerate wandb

# GPU環境の確認
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 3. データセット準備

### 3.1 データセットのフォーマット

VyvoTTSのデータセットには以下のフィールドが必要です：

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `audio` | Audio | 音声データ（WAV形式推奨） |
| `text` | String | 対応するテキスト（日本語） |
| `source`（オプション） | String | 話者名（多話者の場合） |

### 3.2 データセットの作成例

```python
from datasets import Dataset, Audio
import os

def create_dataset_from_folder(audio_folder, transcript_file):
    """
    フォルダから日本語データセットを作成

    Args:
        audio_folder: 音声ファイルのフォルダパス
        transcript_file: テキストファイル（ファイル名\tテキスト形式）
    """
    data = []

    # トランスクリプト読み込み
    with open(transcript_file, "r", encoding="utf-8") as f:
        for line in f:
            filename, text = line.strip().split("\t")
            audio_path = os.path.join(audio_folder, filename)
            if os.path.exists(audio_path):
                data.append({
                    "audio": audio_path,
                    "text": text
                })

    # Dataset作成
    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))

    return dataset

# 使用例
dataset = create_dataset_from_folder(
    audio_folder="./jsut_basic5000/wav",
    transcript_file="./jsut_basic5000/transcript.txt"
)

# HuggingFaceにアップロード
dataset.push_to_hub("your-username/jsut-japanese-tts")
```

### 3.3 音声のトークン化

```python
from vyvotts.audio_tokenizer import process_dataset

# 日本語データセットをトークン化
process_dataset(
    original_dataset="your-username/jsut-japanese-tts",
    output_dataset="your-username/jsut-japanese-tts-tokenized-lfm2",
    model_type="lfm2",  # または "qwen3"
    text_field="text"
)
```

**注意**: トークン化にはCUDA対応GPUが必要です。

---

## 4. ファインチューニング設定

### 4.1 設定ファイルの作成

`vyvotts/configs/train/lfm2_ft_japanese.yaml`を作成：

```yaml
# 日本語データセット（トークン化済み）
TTS_dataset: "your-username/jsut-japanese-tts-tokenized-lfm2"

# ベースモデル
model_name: "Vyvo/VyvoTTS-LFM2-350M-PT"

# Training Args
epochs: 3
batch_size: 8  # VRAMに応じて調整
number_processes: 1
pad_token: 64407
save_steps: 1000
learning_rate: 5.0e-5

# Naming and paths
save_folder: "checkpoints-japanese"
project_name: "vyvotts-japanese"
run_name: "jsut-finetune"
```

### 4.2 GPU別の推奨設定

| GPU | batch_size | gradient_accumulation |
|-----|------------|----------------------|
| RTX 4090（24GB） | 16 | 1 |
| RTX 4070 Ti（16GB） | 8 | 2 |
| T4（16GB） | 4 | 4 |

---

## 5. 学習の実行

### 5.1 単一GPU

```bash
# 設定ファイルのパスを変更してから実行
python vyvotts/train/finetune/train.py
```

### 5.2 マルチGPU（分散学習）

```bash
# accelerateを使用した分散学習
accelerate launch \
    --config_file vyvotts/configs/train/accelerate_finetune.yaml \
    vyvotts/train/finetune/train.py
```

### 5.3 学習の監視

WandBでリアルタイム監視：
```bash
# WandBにログイン
wandb login

# ダッシュボードで確認
# https://wandb.ai/your-username/vyvotts-japanese
```

---

## 6. 推論テスト

### 6.1 学習済みモデルでの推論

```python
import torch
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

# モデルの読み込み
model_path = "./checkpoints-japanese"  # または HuggingFaceのパス
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")

# SNACデコーダー
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

# 日本語テキストで推論
text = "こんにちは、これはテストです。"

# プロンプト構築
prompt = f"<|START_OF_HUMAN|>{text}<|END_OF_TEXT|><|END_OF_HUMAN|><|START_OF_AI|><|START_OF_SPEECH|>"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

# 生成
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=1200,
        temperature=0.6,
        top_p=0.95,
        do_sample=True
    )

# 音声デコード（実装は推論スクリプト参照）
# audio = decode_audio(output, snac)
# sf.write("output_japanese.wav", audio, 24000)

print("推論完了")
```

---

## 7. トラブルシューティング

### よくある問題

| 問題 | 原因 | 解決策 |
|------|------|--------|
| CUDA out of memory | バッチサイズが大きい | batch_sizeを減らす |
| 日本語が文字化け | トークナイザーの問題 | LFM2/Qwen3を使用（日本語対応済み） |
| 音声が不自然 | データ不足 | データ量を増やす（最低3時間推奨） |
| 学習が収束しない | 学習率が高い | learning_rateを1e-5に下げる |

### 日本語特有の注意点

1. **トークン効率**: 日本語は英語の約2倍のトークンが必要
2. **音韻体系**: ピッチアクセント・モーラは明示的にモデル化されていない
3. **句読点**: 「。」「、」を適切に含める

---

## 8. 推定学習時間

### RTX 4090での見積もり

| データ量 | エポック数 | 推定時間 |
|----------|-----------|----------|
| 3時間（JSUT一部） | 3 | 30分〜1時間 |
| 10時間（JSUT全体） | 3 | 2〜4時間 |
| 30時間（JVS） | 3 | 6〜12時間 |

### T4 x 4での見積もり

| データ量 | エポック数 | 推定時間 |
|----------|-----------|----------|
| 3時間（JSUT一部） | 3 | 20〜40分 |
| 10時間（JSUT全体） | 3 | 1.5〜3時間 |
| 30時間（JVS） | 3 | 5〜10時間 |

---

## 9. 次のステップ

1. **品質向上**: より多くのデータで学習
2. **多話者対応**: JVSデータセットで複数話者を学習
3. **感情制御**: 感情ラベル付きデータ（JVNV）で学習
4. **モデル公開**: HuggingFaceにアップロード

```bash
# HuggingFaceにモデルをアップロード
huggingface-cli upload your-username/vyvotts-japanese ./checkpoints-japanese
```

---

## 参考リンク

- [VyvoTTS GitHub](https://github.com/Vyvo-Labs/VyvoTTS)
- [JSUT Corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
- [JVS Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
- [LiquidAI LFM2](https://huggingface.co/LiquidAI/LFM2-350M)
- [SNAC Codec](https://huggingface.co/hubertsiuzdak/snac_24khz)
