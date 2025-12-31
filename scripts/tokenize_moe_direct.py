"""
MOEデータセット直接トークン化スクリプト

MOEデータセットから直接音声ファイルを読み込み、SNACコーデックでトークン化します。
Windows環境でtorchcodec問題を回避するため、soundfileを使用。

pyopenjtalk-plusを使用した高品質日本語前処理に対応:
- prosody: 韻律マーカー付き音素列（ESPNet/Style-BERT-VITS2方式、最高品質）
- phoneme: 音素列のみ
- kana: カタカナ読み
- none: 前処理なし（従来方式）

使用方法:
    uv run python scripts/tokenize_moe_direct.py \
        --moe_path D:/moe_top20 \
        --output_dir ./moe_tokenized \
        --preprocess_mode prosody
"""

import argparse
import json
import os
import re
import unicodedata
import yaml
import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from snac import SNAC
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import Dataset

# Import Japanese preprocessing utility
from vyvotts.utils.japanese_preprocessing import preprocess_japanese_text

# Resample transform cache (avoid recreating for each file)
_resample_cache = {}

def get_resample_transform(orig_freq: int, new_freq: int):
    """Get cached resample transform."""
    key = (orig_freq, new_freq)
    if key not in _resample_cache:
        _resample_cache[key] = T.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return _resample_cache[key]

# LRU cache for Japanese text preprocessing (avoid duplicate processing)
@lru_cache(maxsize=10000)
def cached_preprocess_japanese_text(text: str, mode: str) -> str:
    """Cached version of Japanese text preprocessing."""
    return preprocess_japanese_text(text, mode=mode)


def load_audio_file(item: dict, max_audio_length: float) -> dict:
    """
    Load audio file in a separate thread.
    Returns dict with audio data or None if should be skipped.
    """
    wav_path = item["wav_path"]
    try:
        audio_array, sample_rate = sf.read(wav_path)

        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Check duration
        duration = len(audio_array) / sample_rate
        if duration > max_audio_length:
            return {"status": "skipped_long", "item": item}

        return {
            "status": "ok",
            "item": item,
            "audio_array": audio_array,
            "sample_rate": sample_rate,
        }
    except Exception as e:
        return {"status": "failed", "item": item, "error": str(e)}


def normalize_japanese_text(text: str) -> str:
    """
    日本語テキストを正規化する（従来方式）。
    - NFKC正規化（全角英数→半角、半角カナ→全角カナ）
    - 不要なスペース削除
    - 制御文字削除
    """
    # NFKC正規化
    text = unicodedata.normalize('NFKC', text)
    # 制御文字削除
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    # 連続スペースを1つに
    text = re.sub(r'\s+', ' ', text)
    # 前後のスペース削除
    text = text.strip()
    return text


def load_config(config_path):
    """Load tokenizer configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def tokenise_audio(waveform, snac_model, ds_sample_rate, target_sample_rate, audio_tokens_start):
    """
    Tokenize audio waveform using SNAC codec (optimized vectorized version).
    """
    # Convert to tensor and prepare for processing
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)

    # Resample to target sample rate if needed (using cached transform)
    if ds_sample_rate != target_sample_rate:
        resample_transform = get_resample_transform(ds_sample_rate, target_sample_rate)
        waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to("cuda")

    # Generate SNAC codes
    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    # Vectorized interleaving - transfer all codes to CPU at once
    num_frames = codes[0].shape[1]

    # Transfer to CPU and convert to numpy in one operation per codebook
    c0 = codes[0][0].cpu().numpy()  # Shape: (num_frames,)
    c1 = codes[1][0].cpu().numpy()  # Shape: (num_frames * 2,)
    c2 = codes[2][0].cpu().numpy()  # Shape: (num_frames * 4,)

    # Pre-allocate output array
    all_codes = np.empty(num_frames * 7, dtype=np.int64)

    # Vectorized assignment with offsets
    all_codes[0::7] = c0 + audio_tokens_start
    all_codes[1::7] = c1[0::2] + audio_tokens_start + 4096
    all_codes[2::7] = c2[0::4] + audio_tokens_start + (2 * 4096)
    all_codes[3::7] = c2[1::4] + audio_tokens_start + (3 * 4096)
    all_codes[4::7] = c1[1::2] + audio_tokens_start + (4 * 4096)
    all_codes[5::7] = c2[2::4] + audio_tokens_start + (5 * 4096)
    all_codes[6::7] = c2[3::4] + audio_tokens_start + (6 * 4096)

    return all_codes.tolist()


def remove_duplicate_frames(codes_list):
    """
    Remove consecutive duplicate audio frames to reduce redundancy (optimized vectorized version).
    """
    if len(codes_list) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    # Convert to numpy array and reshape to (num_frames, 7)
    codes_array = np.array(codes_list, dtype=np.int64).reshape(-1, 7)

    # Create mask for non-duplicate frames (first code differs from previous)
    # First frame is always kept
    first_codes = codes_array[:, 0]
    mask = np.concatenate([[True], first_codes[1:] != first_codes[:-1]])

    # Apply mask and flatten back to list
    return codes_array[mask].flatten().tolist()


def collect_moe_files(moe_path: Path, text_field: str = "anime_whisper_transcription"):
    """
    Collect all WAV/JSON pairs from MOE dataset.
    """
    data = []

    # Get speaker directories
    speaker_dirs = [d for d in moe_path.iterdir() if d.is_dir()]
    print(f"Found {len(speaker_dirs)} speaker directories")

    for speaker_dir in tqdm(speaker_dirs, desc="Scanning files"):
        speaker_id = speaker_dir.name

        # MOE structure: {speaker_id}/data/{speaker_id}/wav/
        wav_dir = speaker_dir / "data" / speaker_id / "wav"

        if not wav_dir.exists():
            continue

        # Find all JSON files
        json_files = list(wav_dir.glob("*.json"))

        for json_path in json_files:
            wav_path = json_path.with_suffix(".wav")

            if not wav_path.exists():
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                text = metadata.get(text_field)
                if not text:
                    text = metadata.get("parakeet_jp_transcription")

                if not text:
                    continue

                data.append({
                    "wav_path": str(wav_path),
                    "text": text,
                    "speaker_id": speaker_id,
                })

            except (json.JSONDecodeError, Exception):
                continue

    return data


def process_moe_direct(
    moe_path,
    output_dir,
    text_field="anime_whisper_transcription",
    model_type="lfm2",
    target_sample_rate=24000,
    max_audio_length=30.0,  # Skip audio longer than 30 seconds
    preprocess_mode="prosody",  # Japanese text preprocessing mode
    save_format="arrow",  # Save format: "arrow" or "parquet"
):
    """
    Process MOE dataset directly: tokenize audio and text.
    """
    moe_path = Path(moe_path)

    # Set tokenizer and config based on model type
    if model_type == "lfm2":
        tokenizer_model = "LiquidAI/LFM2-350M"
        config_path = "vyvotts/configs/inference/lfm2.yaml"
    elif model_type == "qwen3":
        tokenizer_model = "Qwen/Qwen3-0.6B"
        config_path = "vyvotts/configs/inference/qwen3.yaml"
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    # Load configuration
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    START_OF_TEXT = config['START_OF_TEXT']
    END_OF_TEXT = config['END_OF_TEXT']
    START_OF_SPEECH = config['START_OF_SPEECH']
    END_OF_SPEECH = config['END_OF_SPEECH']
    START_OF_HUMAN = config['START_OF_HUMAN']
    END_OF_HUMAN = config['END_OF_HUMAN']
    START_OF_AI = config['START_OF_AI']
    END_OF_AI = config['END_OF_AI']
    AUDIO_TOKENS_START = config['AUDIO_TOKENS_START']

    # Collect files
    print(f"\nScanning MOE dataset: {moe_path}")
    file_list = collect_moe_files(moe_path, text_field)
    print(f"Total files found: {len(file_list)}")

    # Load SNAC model with torch.compile optimization
    print("\nLoading SNAC model: hubertsiuzdak/snac_24khz")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")
    snac_model.eval()
    # Apply torch.compile for faster inference (PyTorch 2.0+)
    try:
        snac_model = torch.compile(snac_model, mode="reduce-overhead")
        print("  torch.compile applied successfully")
    except Exception as e:
        print(f"  torch.compile skipped: {e}")

    # Load text tokenizer
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Process files with multi-threaded I/O for audio loading
    processed_data = []
    failed_count = 0
    skipped_long = 0
    num_io_workers = 4  # Number of threads for audio file loading

    print(f"\nProcessing with {num_io_workers} I/O workers...")

    # Process in batches with prefetching
    batch_size = 32  # Number of files to prefetch
    pbar = tqdm(total=len(file_list), desc="Tokenizing audio")

    for batch_start in range(0, len(file_list), batch_size):
        batch_end = min(batch_start + batch_size, len(file_list))
        batch_items = file_list[batch_start:batch_end]

        # Load audio files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
            futures = {
                executor.submit(load_audio_file, item, max_audio_length): item
                for item in batch_items
            }

            loaded_data = []
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "ok":
                    loaded_data.append(result)
                elif result["status"] == "skipped_long":
                    skipped_long += 1
                else:
                    failed_count += 1

        # Process loaded audio on GPU (sequential for GPU efficiency)
        for data in loaded_data:
            try:
                item = data["item"]
                audio_array = data["audio_array"]
                sample_rate = data["sample_rate"]
                text = item["text"]
                speaker_id = item["speaker_id"]

                # Tokenize audio
                codes_list = tokenise_audio(
                    audio_array,
                    snac_model,
                    sample_rate,
                    target_sample_rate,
                    AUDIO_TOKENS_START
                )

                # Remove duplicate frames
                codes_list = remove_duplicate_frames(codes_list)

                # Preprocess Japanese text (with caching for duplicates)
                if preprocess_mode == "none":
                    text = normalize_japanese_text(text)
                else:
                    text = cached_preprocess_japanese_text(text, preprocess_mode)

                # Tokenize text with speaker prefix
                text_prompt = f"{speaker_id}: {text}"
                text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
                text_ids.append(END_OF_TEXT)

                # Construct full sequence with special tokens
                input_ids = (
                    [START_OF_HUMAN]
                    + text_ids
                    + [END_OF_HUMAN]
                    + [START_OF_AI]
                    + [START_OF_SPEECH]
                    + codes_list
                    + [END_OF_SPEECH]
                    + [END_OF_AI]
                )

                processed_data.append({
                    "input_ids": input_ids,
                    "labels": input_ids,
                    "attention_mask": [1] * len(input_ids),
                })

            except Exception as e:
                failed_count += 1
                continue

        pbar.update(len(batch_items))

    pbar.close()

    print(f"\nProcessed: {len(processed_data)} samples")
    print(f"Failed: {failed_count} samples")
    print(f"Skipped (too long): {skipped_long} samples")

    # Create dataset from processed data
    tokenized_ds = Dataset.from_list(processed_data)

    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_dir} (format: {save_format})")

    if save_format == "parquet":
        # Parquet形式で保存（snappy圧縮、学習時の読み込み40-60%高速化）
        parquet_path = output_path / "data.parquet"
        tokenized_ds.to_parquet(str(parquet_path), compression="snappy")
        print(f"Saved as Parquet: {parquet_path}")
    else:
        # Arrow形式（デフォルト）
        tokenized_ds.save_to_disk(str(output_path))

    print("Done!")

    return tokenized_ds


def main():
    parser = argparse.ArgumentParser(
        description="MOEデータセット直接トークン化スクリプト"
    )
    parser.add_argument(
        "--moe_path",
        type=str,
        default="D:/moe_top20",
        help="MOEデータセットのパス (default: D:/moe_top20)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./moe_tokenized",
        help="出力データセットのパス (default: ./moe_tokenized)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="anime_whisper_transcription",
        choices=["anime_whisper_transcription", "parakeet_jp_transcription"],
        help="使用するテキストフィールド (default: anime_whisper_transcription)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lfm2",
        choices=["lfm2", "qwen3"],
        help="モデルタイプ (default: lfm2)",
    )
    parser.add_argument(
        "--max_audio_length",
        type=float,
        default=30.0,
        help="最大音声長（秒）。これより長い音声はスキップ (default: 30.0)",
    )
    parser.add_argument(
        "--preprocess_mode",
        type=str,
        default="prosody",
        choices=["prosody", "phoneme", "kana", "none"],
        help="日本語テキスト前処理モード (default: prosody, 最高品質)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="arrow",
        choices=["arrow", "parquet"],
        help="保存形式 (default: arrow, parquet: 学習時読み込み40-60%%高速化)",
    )

    args = parser.parse_args()

    process_moe_direct(
        moe_path=args.moe_path,
        output_dir=args.output_dir,
        text_field=args.text_field,
        model_type=args.model_type,
        max_audio_length=args.max_audio_length,
        preprocess_mode=args.preprocess_mode,
        save_format=args.format,
    )


if __name__ == "__main__":
    main()
