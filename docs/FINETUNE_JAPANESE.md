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

#### ローカルデータセット（MOE等）の場合

```bash
# Parquet形式（デフォルト、学習時読み込み40-60%高速化）
uv run python scripts/tokenize_moe_direct.py \
    --moe_path D:/moe_top20 \
    --output_dir ./moe_tokenized \
    --model_type lfm2 \
    --preprocess_mode prosody

# Arrow形式（従来方式）
uv run python scripts/tokenize_moe_direct.py \
    --moe_path D:/moe_top20 \
    --output_dir ./moe_tokenized \
    --format arrow
```

#### HuggingFaceデータセットの場合

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

#### データセット形式

| 形式 | 説明 | 用途 |
|------|------|------|
| **Parquet** (デフォルト) | Snappy圧縮、高速読み込み | 推奨（学習時40-60%高速化） |
| Arrow | HuggingFace標準形式 | 互換性重視 |

学習スクリプトはParquet/Arrow形式を自動検出するため、`--format`オプションを意識する必要はありません。

---

## 4. ファインチューニング設定

### 4.1 設定ファイルの作成

既存の設定ファイルをコピーして日本語用に編集します：

```bash
# 既存ファイルをコピー
cp vyvotts/configs/train/lfm2_ft.yaml vyvotts/configs/train/lfm2_ft_japanese.yaml
```

`vyvotts/configs/train/lfm2_ft_japanese.yaml`を以下のように編集：

```yaml
# 日本語データセット（トークン化済み）
TTS_dataset: "your-username/jsut-japanese-tts-tokenized-lfm2"

# ベースモデル
model_name: "Vyvo/VyvoTTS-LFM2-350M-PT"

# Training Args
epochs: 3
batch_size: 8  # VRAMに応じて調整（下記参照）
number_processes: 1
pad_token: 64407
save_steps: 1000
learning_rate: 5.0e-5

# Naming and paths
save_folder: "checkpoints-japanese"
project_name: "vyvotts-japanese"
run_name: "jsut-finetune"
```

**重要**: 学習スクリプト実行前に、`train.py`内の設定ファイルパスを更新するか、環境変数で指定してください。

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

### 6.1 学習済みモデルでの推論（推奨）

既存の推論クラスを使用する方法（シンプル）：

```python
from vyvotts.inference.transformers_inference import VyvoTTSTransformersInference

# 学習済みモデルで推論エンジンを初期化
engine = VyvoTTSTransformersInference(
    model_name="./checkpoints-japanese"  # または HuggingFaceのパス
)

# 日本語テキストで推論
audio, timing = engine.generate(
    text="こんにちは、これはテストです。",
    output_path="output_japanese.wav"
)

print(f"推論完了: {timing['total_time']:.2f}秒")
print(f"音声ファイル: output_japanese.wav")
```

### 6.2 手動での推論（詳細制御が必要な場合）

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

# モデルの読み込み
model_path = "./checkpoints-japanese"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cuda")

# 詳細な推論実装については vyvotts/inference/transformers_inference.py を参照
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

## 8. 日本語テキスト前処理（高品質モード）

### pyopenjtalk-plus による韻律処理

VyvoTTSは `pyopenjtalk-plus` を使用した高品質な日本語前処理をサポートしています。これはESPNet/Style-BERT-VITS2で採用されている方式です。

#### 前処理モード

| モード | 説明 | 品質 |
|--------|------|------|
| `prosody` | 韻律マーカー付き音素列（推奨） | 最高 |
| `phoneme` | 音素列のみ | 良好 |
| `kana` | カタカナ読み | 標準 |
| `none` | 前処理なし（従来方式） | 低 |

#### 韻律マーカーの意味

```
入力: "こんにちは、今日はいい天気ですね。"
出力: "^ k o [ N n i ch i w a # k y o o w a [ i i t e N k i d e s u n e $"
```

| 記号 | 意味 |
|------|------|
| `^` | 文頭 |
| `$` | 文末（平叙文） |
| `?` | 文末（疑問文） |
| `#` | アクセント句境界 |
| `[` | ピッチ上昇 |
| `]` | ピッチ下降 |

#### 使用例

```python
from vyvotts.utils.japanese_preprocessing import preprocess_japanese_text

# 韻律マーカー付き（最高品質）
text = preprocess_japanese_text("音声合成技術", mode="prosody")
# → "^ [ o N s e e g o o s e e g i j u ts u $"

# 音素列のみ
text = preprocess_japanese_text("音声合成技術", mode="phoneme")
# → "o N s e i g o o s e i g i j u ts u"
```

---

## 9. パフォーマンス最適化

### トークン化の最適化

以下の最適化により、トークン化処理が**約60%高速化**されています。

| 最適化 | 効果 | 説明 |
|--------|------|------|
| `.item()`ループのベクトル化 | 10-50x | GPU-CPU転送を一括化 |
| 重複フレーム検出のベクトル化 | 10-50x | numpy配列操作に変更 |
| Resampleトランスフォームキャッシュ | 軽微 | オブジェクトの再利用 |
| 日本語前処理LRUキャッシュ | 2-5x | 重複テキストの処理削減 |
| torch.compile (SNAC) | 10-20% | GPUカーネル最適化 |
| マルチスレッドI/O | 2-3x | ThreadPoolExecutorで並列読み込み |
| **Parquet形式** (デフォルト) | 40-60% | 学習時データ読み込み高速化 |

#### 実行結果

| 項目 | 最適化前 | 最適化後 |
|------|---------|---------|
| 処理速度 | ~17 items/sec | ~27 items/sec |
| 53K件の処理時間 | ~50分 | ~33分 |

### 学習の最適化

以下の最適化により、学習処理が**約35-50%高速化**されています。

| 最適化 | 効果 | 説明 |
|--------|------|------|
| `dataloader_num_workers=4` | 20-30% | データローディング並列化 |
| `dataloader_pin_memory=True` | 5-10% | CPU→GPU転送高速化 |
| `dataloader_prefetch_factor=2` | 5-10% | バッチプリフェッチ |
| `logging_steps=50` | 3-5% | ログ頻度削減（10→50） |
| データコレータ最適化 | 5-10% | `pad_sequence`使用 |
| torch.compile (モデル) | 15-30% | PyTorch 2.0+最適化 |

#### torch.compileの使用

```bash
# torch.compileを有効にして学習（デフォルト）
uv run python scripts/train_japanese.py --dataset_path ./moe_tokenized_prosody

# torch.compileを無効にして学習
uv run python scripts/train_japanese.py --dataset_path ./moe_tokenized_prosody --no_compile
```

---

## 10. 推定学習時間（最適化適用後）

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

## 11. 次のステップ

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
