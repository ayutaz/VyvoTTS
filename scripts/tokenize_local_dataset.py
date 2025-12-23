"""
ローカルデータセット音声トークン化スクリプト

ローカルのHuggingFace Datasetを読み込み、SNACコーデックで音声をトークン化します。

使用方法:
    uv run python scripts/tokenize_local_dataset.py \
        --input_dir ./jsut_dataset \
        --output_dir ./jsut_tokenized \
        --model_type lfm2
"""

import argparse
import os
import yaml
import torch
import torchaudio.transforms as T
from datasets import load_from_disk
from snac import SNAC
from transformers import AutoTokenizer
from tqdm import tqdm


def load_config(config_path):
    """Load tokenizer configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def tokenise_audio(waveform, snac_model, ds_sample_rate, target_sample_rate, audio_tokens_start):
    """
    Tokenize audio waveform using SNAC codec.

    Args:
        waveform: Audio array from dataset
        snac_model: SNAC model instance
        ds_sample_rate: Original dataset sample rate
        target_sample_rate: Target sample rate (24000)
        audio_tokens_start: Offset for audio tokens

    Returns:
        List of audio token IDs with proper offsets applied
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


def process_local_dataset(
    input_dir,
    output_dir,
    model_type="lfm2",
    text_field="text",
    target_sample_rate=24000
):
    """
    Process local dataset: tokenize audio and text, create training sequences.

    Args:
        input_dir: Path to local HuggingFace dataset
        output_dir: Path to save tokenized dataset
        model_type: Model type - either "qwen3" or "lfm2" (default: "lfm2")
        text_field: Name of text field in dataset (default: "text")
        target_sample_rate: Target audio sample rate (default: 24000)
    """
    # Set tokenizer and config based on model type
    if model_type == "qwen3":
        tokenizer_model = "Qwen/Qwen3-0.6B"
        config_path = "vyvotts/configs/inference/qwen3.yaml"
    elif model_type == "lfm2":
        tokenizer_model = "LiquidAI/LFM2-350M"
        config_path = "vyvotts/configs/inference/lfm2.yaml"
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'qwen3' or 'lfm2'")

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

    # Load local dataset
    print(f"Loading dataset from: {input_dir}")
    ds = load_from_disk(input_dir)
    print(f"Dataset loaded: {len(ds)} samples")

    # Get sample rate from first audio sample
    ds_sample_rate = ds[0]["audio"]["sampling_rate"]
    print(f"Source sample rate: {ds_sample_rate}")
    print(f"Target sample rate: {target_sample_rate}")

    # Load SNAC model
    print("Loading SNAC model: hubertsiuzdak/snac_24khz")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")

    # Load text tokenizer
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Process each sample
    processed_data = []
    failed_count = 0

    for idx in tqdm(range(len(ds)), desc="Tokenizing audio"):
        example = ds[idx]

        try:
            audio_data = example.get("audio")
            if audio_data is None or "array" not in audio_data:
                failed_count += 1
                continue

            audio_array = audio_data["array"]

            # Tokenize audio
            codes_list = tokenise_audio(
                audio_array,
                snac_model,
                ds_sample_rate,
                target_sample_rate,
                AUDIO_TOKENS_START
            )

            # Remove duplicate frames
            codes_list = remove_duplicate_frames(codes_list)

            # Tokenize text
            text_prompt = example[text_field]
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
            print(f"Error processing sample {idx}: {e}")
            failed_count += 1
            continue

    print(f"\nProcessed: {len(processed_data)} samples")
    print(f"Failed: {failed_count} samples")

    # Create dataset from processed data
    from datasets import Dataset
    tokenized_ds = Dataset.from_list(processed_data)

    # Save dataset
    print(f"Saving to: {output_dir}")
    tokenized_ds.save_to_disk(output_dir)
    print("Done!")

    return tokenized_ds


def main():
    parser = argparse.ArgumentParser(
        description="ローカルデータセット音声トークン化スクリプト"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="入力データセットのパス (例: ./jsut_dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="出力データセットのパス (例: ./jsut_tokenized)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lfm2",
        choices=["lfm2", "qwen3"],
        help="モデルタイプ (default: lfm2)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="テキストフィールド名 (default: text)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="目標サンプルレート (default: 24000)",
    )

    args = parser.parse_args()

    process_local_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        text_field=args.text_field,
        target_sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
