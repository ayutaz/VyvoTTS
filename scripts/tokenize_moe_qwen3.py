"""
MOEデータセット Qwen3用トークン化スクリプト

Qwen3モデル向けにMOEデータセットをトークン化します。
Qwen3はvocab size 151,669で日本語トークン効率が向上します。

使用方法:
    uv run python scripts/tokenize_moe_qwen3.py \
        --moe_path D:/moe_top20 \
        --output_dir ./moe_tokenized_qwen3
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
from pathlib import Path
from snac import SNAC
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import Dataset


def normalize_japanese_text(text: str) -> str:
    """
    日本語テキストを正規化する。
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
    Tokenize audio waveform using SNAC codec.
    """
    # Convert to tensor and prepare for processing
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)

    # Resample to target sample rate if needed
    if ds_sample_rate != target_sample_rate:
        resample_transform = T.Resample(orig_freq=ds_sample_rate, new_freq=target_sample_rate)
        waveform = resample_transform(waveform)

    waveform = waveform.unsqueeze(0).to("cuda")

    # Generate SNAC codes
    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    # Interleave codes from 3 codebooks with proper offsets
    all_codes = []
    num_frames = codes[0].shape[1]

    for i in range(num_frames):
        # Level 0: 1 code per frame
        all_codes.append(codes[0][0][i].item() + audio_tokens_start)
        # Level 1: 2 codes per frame
        all_codes.append(codes[1][0][2*i].item() + audio_tokens_start + 4096)
        # Level 2: 4 codes per frame
        all_codes.append(codes[2][0][4*i].item() + audio_tokens_start + (2 * 4096))
        all_codes.append(codes[2][0][4*i + 1].item() + audio_tokens_start + (3 * 4096))
        # Continue level 1 and 2 interleaving
        all_codes.append(codes[1][0][2*i + 1].item() + audio_tokens_start + (4 * 4096))
        all_codes.append(codes[2][0][4*i + 2].item() + audio_tokens_start + (5 * 4096))
        all_codes.append(codes[2][0][4*i + 3].item() + audio_tokens_start + (6 * 4096))

    return all_codes


def remove_duplicate_frames(codes_list):
    """
    Remove consecutive duplicate audio frames to reduce redundancy.
    """
    if len(codes_list) % 7 != 0:
        raise ValueError("Input list length must be divisible by 7")

    result = codes_list[:7]

    for i in range(7, len(codes_list), 7):
        current_first_code = codes_list[i]
        previous_first_code = result[-7]

        if current_first_code != previous_first_code:
            result.extend(codes_list[i:i+7])

    return result


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


def process_moe_qwen3(
    moe_path,
    output_dir,
    text_field="anime_whisper_transcription",
    target_sample_rate=24000,
    max_audio_length=30.0,
):
    """
    Process MOE dataset for Qwen3: tokenize audio and text.
    """
    moe_path = Path(moe_path)

    # Qwen3 configuration
    tokenizer_model = "Qwen/Qwen3-0.6B"
    config_path = "vyvotts/configs/inference/qwen3.yaml"

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

    print(f"Qwen3 config loaded:")
    print(f"  AUDIO_TOKENS_START: {AUDIO_TOKENS_START}")
    print(f"  Tokenizer: {tokenizer_model}")

    # Collect files
    print(f"\nScanning MOE dataset: {moe_path}")
    file_list = collect_moe_files(moe_path, text_field)
    print(f"Total files found: {len(file_list)}")

    # Load SNAC model
    print("\nLoading SNAC model: hubertsiuzdak/snac_24khz")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")

    # Load Qwen3 tokenizer
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Process each file
    processed_data = []
    failed_count = 0
    skipped_long = 0

    for item in tqdm(file_list, desc="Tokenizing audio"):
        wav_path = item["wav_path"]
        text = item["text"]
        speaker_id = item["speaker_id"]

        try:
            # Load audio with soundfile
            audio_array, sample_rate = sf.read(wav_path)

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            # Skip very long audio
            duration = len(audio_array) / sample_rate
            if duration > max_audio_length:
                skipped_long += 1
                continue

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

            # Normalize Japanese text
            text = normalize_japanese_text(text)

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

    print(f"\nProcessed: {len(processed_data)} samples")
    print(f"Failed: {failed_count} samples")
    print(f"Skipped (too long): {skipped_long} samples")

    # Create dataset from processed data
    tokenized_ds = Dataset.from_list(processed_data)

    # Save dataset
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {output_dir}")
    tokenized_ds.save_to_disk(str(output_path))
    print("Done!")

    return tokenized_ds


def main():
    parser = argparse.ArgumentParser(
        description="MOEデータセット Qwen3用トークン化スクリプト"
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
        default="./moe_tokenized_qwen3",
        help="出力データセットのパス (default: ./moe_tokenized_qwen3)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="anime_whisper_transcription",
        choices=["anime_whisper_transcription", "parakeet_jp_transcription"],
        help="使用するテキストフィールド (default: anime_whisper_transcription)",
    )
    parser.add_argument(
        "--max_audio_length",
        type=float,
        default=30.0,
        help="最大音声長（秒）。これより長い音声はスキップ (default: 30.0)",
    )

    args = parser.parse_args()

    process_moe_qwen3(
        moe_path=args.moe_path,
        output_dir=args.output_dir,
        text_field=args.text_field,
        max_audio_length=args.max_audio_length,
    )


if __name__ == "__main__":
    main()
