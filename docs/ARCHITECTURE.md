# VyvoTTS アーキテクチャと特徴

## 概要

VyvoTTSは**LLM（大規模言語モデル）ベースのText-to-Speech（TTS）フレームワーク**です。

- [Orpheus TTS](https://github.com/canopyai/orpheus-tts)をベースに開発
- テキスト→音声トークン→音声波形の自己回帰生成
- 軽量モデル（350M-600M）で高品質な音声合成を実現

## アーキテクチャ

### データフロー

```
テキスト入力
    ↓
[特殊トークンでラップ]
[START_OF_HUMAN] テキスト [END_OF_TEXT] [END_OF_HUMAN]
    ↓
[LLMモデル（LFM2 / Qwen3）]
音声トークンを自己回帰的に生成
    ↓
[音声トークン列]
7コード/フレーム形式
    ↓
[SNACデコーダー]
音声トークン → 音声波形
    ↓
24kHz WAV音声出力
```

### SNACコーデック

VyvoTTSは[SNAC（Scalable Neural Audio Codec）](https://huggingface.co/hubertsiuzdak/snac_24khz)を使用しています。

#### 3階層ベクトル量子化

| 階層 | コード数/フレーム | 役割 |
|------|------------------|------|
| Layer 0 | 1 | 粗い音声特徴（基本周波数等） |
| Layer 1 | 2 | 中間詳細 |
| Layer 2 | 4 | 細かい音声詳細（音質） |
| **合計** | **7** | 1フレームあたり |

#### インターリーブ配置

```
フレーム i のコード配置:
[L0] [L1a] [L2a] [L2b] [L1b] [L2c] [L2d]
  ↓    ↓     ↓     ↓     ↓     ↓     ↓
+0  +4096 +8192 +12288 +16384 +20480 +24576
```

各コードは`AUDIO_TOKENS_START`（64410）からのオフセットで管理されます。

#### フレーム重複削除

連続する同一フレームを自動削除し、シーケンス長を短縮します。

### トークン構造

```
入力形式:
[START_OF_HUMAN] テキスト [END_OF_TEXT] [END_OF_HUMAN] [START_OF_AI] [START_OF_SPEECH] 音声コード... [END_OF_SPEECH] [END_OF_AI]
```

特殊トークンはYAML設定（`vyvotts/configs/inference/*.yaml`）で定義されています。

## 技術的特徴

### 1. 軽量LLMモデル

| モデル | パラメータ数 | トークナイザー長 | 特徴 |
|--------|-------------|-----------------|------|
| **LFM2** | 350M | 64,400 | 推奨、高効率 |
| **Qwen3** | 600M | 151,669 | より大きなボキャブラリー |

### 2. GradualRatioDataset（段階的学習）

プリトレーニング時に、テキストQAデータと音声データの比率を段階的に変更します。

```
訓練開始時: テキスト:音声 = 2:1
    ↓ （訓練進行に伴い線形補間）
訓練終了時: テキスト:音声 = 1:1
```

**効果:**
- 初期: テキスト理解能力を強化
- 後期: 音声合成能力へシフト
- 転移学習の効果を最大化

### 3. 複数推論エンジン

| エンジン | 速度 | メモリ | 特徴 |
|----------|------|--------|------|
| **Transformers** | 中 | 高 | 標準、Flash Attention対応 |
| **vLLM** | 最速 | 中 | 本番環境向け、Linux専用 |
| **Unsloth** | 高 | 低 | 4bit/8bit量子化、6GB+ VRAM |
| **HQQ** | 高 | 低 | 高品質量子化（1-8bit） |

### 4. ボイスクローニング

参照音声をコンテキストとして組み込むことで、追加モジュールなしにボイスクローニングを実現します。

```python
from vyvotts.voice_clone import VyvoTTSVoiceClone

cloner = VyvoTTSVoiceClone()
audio = cloner.clone_voice(
    reference_audio_path="reference.wav",
    reference_transcript="参照音声のテキスト",
    target_texts=["合成したいテキスト"]
)
```

## 他のLLMベースTTSとの比較

| モデル | パラメータ | ボイスクローン | 低VRAM | 特徴 |
|--------|-----------|---------------|--------|------|
| **VyvoTTS** | 350M-600M | ✅ | ✅ 6GB+ | 軽量、複数エンジン、柔軟な設定 |
| **Bark** | ~1B | ❌ | ❌ | 音楽・効果音も生成可能、表現力豊か |
| **VALL-E** | ~1B | ✅ | ❌ | ゼロショット音声クローニングの先駆け |
| **Parler-TTS** | - | ❌ | - | 性別・ピッチ・スタイル制御 |
| **Spark-TTS** | - | ✅ | ✅ | 効率的な単一ストリーム（2025年発表） |
| **Kokoro** | 82M | ❌ | ✅ | 超軽量だがクローニング非対応 |
| **NVIDIA T5-TTS** | - | - | - | ハルシネーション対策に優れる |

### VyvoTTSの強み

1. **軽量モデル**: 350Mパラメータで高品質な音声合成
2. **低VRAM対応**: 4bit量子化で6GB+ VRAMから動作
3. **統一パイプライン**: データ準備→訓練→推論が一貫した設計
4. **柔軟なYAML設定**: モデル切り替えが容易
5. **複数推論エンジン**: 用途に応じて選択可能
6. **ボイスクローニング**: 追加モジュール不要

### LLMベースTTSの共通課題

| 課題 | 説明 | VyvoTTSの対応 |
|------|------|--------------|
| ハルシネーション | 単語の繰り返し・欠落・ズレ | repetition_penaltyで軽減 |
| 計算コスト | 自己回帰生成は遅い | 軽量モデル + 量子化 |
| 多言語対応 | 訓練データに依存 | 現在は英語のみ |

## Orpheus TTSからの改善・変更点

VyvoTTSは[Orpheus TTS](https://github.com/canopyai/orpheus-tts)をベースに開発されていますが、以下の改善・変更が行われています。

### 1. モデルサイズの大幅な縮小

| 項目 | Orpheus TTS | VyvoTTS |
|------|-------------|---------|
| ベースモデル | Llama-3.2-3B | LFM2-350M / Qwen3-600M |
| パラメータ数 | **3B** | **350M〜600M**（5〜10倍小さい） |

### 2. 推論エンジンの多様化

Orpheus TTSはvLLMのみ対応ですが、VyvoTTSは用途に応じて4種類のエンジンを選択可能：

| エンジン | 特徴 |
|----------|------|
| **Transformers** | 標準、Flash Attention対応 |
| **vLLM** | 最速（Linux専用） |
| **Unsloth** | 4bit/8bit量子化、省メモリ |
| **HQQ** | 高品質量子化（1-8bit） |

### 3. 低VRAM環境対応

Orpheus TTSは3Bモデルのため高VRAMが必要ですが、VyvoTTSは量子化により**6GB+ VRAMから動作可能**です。

### 4. GradualRatioDataset（段階的学習）

VyvoTTS独自の訓練機構として、プリトレーニング時にテキストQAデータと音声データの比率を段階的に変更する`GradualRatioDataset`を実装しています。

```python
# vyvotts/train/pretrain/train.py
class GradualRatioDataset(Dataset):
    # 訓練進行に伴い、テキスト:音声比率を段階的に変更
    # 初期: 2:1 → 終了: 1:1
```

### 5. 完全な訓練フレームワークの提供

Orpheus TTSは訓練コードを公開していませんが、VyvoTTSは以下を含む完全な訓練パイプラインを提供：

- プリトレーニング（FSDP対応）
- ファインチューニング
- Liger Kernel統合（最適化）
- データセット準備ツール（`audio_tokenizer.py`）

### 6. YAML設定による柔軟な管理

```
vyvotts/configs/
├── inference/
│   ├── lfm2.yaml    # LFM2用トークン定義
│   ├── qwen3.yaml   # Qwen3用トークン定義
│   └── llama3.yaml  # Llama3用トークン定義
└── train/
    ├── lfm2_ft.yaml      # ファインチューン設定
    └── lfm2_pretrain.yaml # プリトレーン設定
```

### 機能比較まとめ

| 機能 | Orpheus | VyvoTTS | 備考 |
|------|---------|---------|------|
| ゼロショット音声クローニング | ✅ | ✅ | 同等 |
| 感情タグ | ✅ `<laugh>` 等 | ❓ | Orpheus優位 |
| 多言語対応 | ✅ 研究版 | ❌ 英語のみ | Orpheus優位 |
| 訓練コード | ❌ | ✅ | VyvoTTS優位 |
| 低VRAM対応 | ❌ | ✅ 6GB+ | VyvoTTS優位 |
| 複数推論エンジン | ❌ | ✅ 4種 | VyvoTTS優位 |
| 軽量モデル | ❌ 3B | ✅ 350M | VyvoTTS優位 |

### アーキテクチャ的新規性について

VyvoTTSはOrpheus TTSの**実装改良版**であり、コアアーキテクチャ（LLM + SNACコーデック）は継承しています。

- **研究的新規性**: 限定的（GradualRatioDatasetはCurriculum Learningの応用）
- **実用的価値**: 高い（軽量化、低VRAM対応、訓練フレームワーク提供）

## 制限事項

- **対応言語**: 英語のみ（訓練データセットが英語）
- **vLLM**: Linux専用（Windowsでは使用不可）
- **Flash Attention 3**: Windows未対応（Flash Attention 2を使用）

## 参考リンク

- [VyvoTTS GitHub](https://github.com/Vyvo-Labs/VyvoTTS)
- [Orpheus TTS](https://github.com/canopyai/orpheus-tts)
- [SNAC Codec](https://huggingface.co/hubertsiuzdak/snac_24khz)
- [LiquidAI LFM2](https://huggingface.co/LiquidAI/LFM2-350M)
