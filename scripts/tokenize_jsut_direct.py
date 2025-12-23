"""
JSUT直接トークン化スクリプト

JSUTフォルダから直接WAVファイルを読み込み、SNACコーデックで音声をトークン化します。

使用方法:
    uv run python scripts/tokenize_jsut_direct.py \
        --jsut_path ./jsut_ver1.1 \
        --output_dir ./jsut_tokenized \
        --model_type lfm2
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
import torchaudio
from tqdm import tqdm
from snac import SNAC
from transformers import AutoTokenizer


# JSUTのサブセット一覧
JSUT_SUBSETS = [
    "basic5000",
    "travel1000",
    "repeat500",
    "utparaphrase512",
    "onomatopee300",
    "precedent130",
    "loanword128",
    "voiceactress100",
    "countersuffix26",
]


def load_config(config_path):
    """Load tokenizer configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_transcript(transcript_path: Path) -> dict:
    """
    トランスクリプトファイルを読み込む
    """
    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) == 2:
                filename, text = parts
                transcripts[filename] = text
    return transcripts


def tokenise_audio(waveform, snac_model, source_sample_rate, target_sample_rate, audio_tokens_start):
    """
    Tokenize audio waveform using SNAC codec.
    """
    # Resample if needed
    if source_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(source_sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Add batch dimension and move to GPU
    waveform = waveform.unsqueeze(0).to("cuda")

    # Generate SNAC codes
    with torch.inference_mode():
        codes = snac_model.encode(waveform)

    # Interleave codes from 3 codebooks with proper offsets
    all_codes = []
    num_frames = codes[0].shape[1]

    for i in range(num_frames):
        all_codes.append(codes[0][0][i].item() + audio_tokens_start)
        all_codes.append(codes[1][0][2*i].item() + audio_tokens_start + 4096)
        all_codes.append(codes[2][0][4*i].item() + audio_tokens_start + (2 * 4096))
        all_codes.append(codes[2][0][4*i + 1].item() + audio_tokens_start + (3 * 4096))
        all_codes.append(codes[1][0][2*i + 1].item() + audio_tokens_start + (4 * 4096))
        all_codes.append(codes[2][0][4*i + 2].item() + audio_tokens_start + (5 * 4096))
        all_codes.append(codes[2][0][4*i + 3].item() + audio_tokens_start + (6 * 4096))

    return all_codes


def remove_duplicate_frames(codes_list):
    """
    Remove consecutive duplicate audio frames.
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


def process_jsut_direct(
    jsut_path,
    output_dir,
    model_type="lfm2",
    subsets=None,
    target_sample_rate=24000
):
    """
    Process JSUT directly from folder.
    """
    jsut_path = Path(jsut_path)
    output_dir = Path(output_dir)

    if subsets is None:
        subsets = JSUT_SUBSETS

    # Set tokenizer and config based on model type
    if model_type == "qwen3":
        tokenizer_model = "Qwen/Qwen3-0.6B"
        config_path = "vyvotts/configs/inference/qwen3.yaml"
    elif model_type == "lfm2":
        tokenizer_model = "LiquidAI/LFM2-350M"
        config_path = "vyvotts/configs/inference/lfm2.yaml"
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    # Load configuration
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    END_OF_TEXT = config['END_OF_TEXT']
    START_OF_SPEECH = config['START_OF_SPEECH']
    END_OF_SPEECH = config['END_OF_SPEECH']
    START_OF_HUMAN = config['START_OF_HUMAN']
    END_OF_HUMAN = config['END_OF_HUMAN']
    START_OF_AI = config['START_OF_AI']
    END_OF_AI = config['END_OF_AI']
    AUDIO_TOKENS_START = config['AUDIO_TOKENS_START']

    # Load SNAC model
    print("Loading SNAC model: hubertsiuzdak/snac_24khz")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to("cuda")
    snac_model.eval()

    # Load text tokenizer
    print(f"Loading tokenizer: {tokenizer_model}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    # Collect all samples
    all_samples = []
    for subset in subsets:
        subset_path = jsut_path / subset
        if not subset_path.exists():
            print(f"Warning: {subset} not found, skipping...")
            continue

        transcript_path = subset_path / "transcript_utf8.txt"
        wav_dir = subset_path / "wav"

        if not transcript_path.exists() or not wav_dir.exists():
            continue

        transcripts = load_transcript(transcript_path)
        for filename, text in transcripts.items():
            wav_path = wav_dir / f"{filename}.wav"
            if wav_path.exists():
                all_samples.append({
                    "wav_path": wav_path,
                    "text": text,
                    "subset": subset,
                })

    print(f"Total samples: {len(all_samples)}")

    # Process each sample
    processed_data = []
    failed_count = 0

    for sample in tqdm(all_samples, desc="Tokenizing audio"):
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(sample["wav_path"])

            # Tokenize audio
            codes_list = tokenise_audio(
                waveform,
                snac_model,
                sample_rate,
                target_sample_rate,
                AUDIO_TOKENS_START
            )

            # Remove duplicate frames
            codes_list = remove_duplicate_frames(codes_list)

            # Tokenize text
            text_ids = tokenizer.encode(sample["text"], add_special_tokens=True)
            text_ids.append(END_OF_TEXT)

            # Construct full sequence
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
            print(f"Error processing {sample['wav_path']}: {e}")
            failed_count += 1
            continue

    print(f"\nProcessed: {len(processed_data)} samples")
    print(f"Failed: {failed_count} samples")

    # Create and save dataset
    from datasets import Dataset
    tokenized_ds = Dataset.from_list(processed_data)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {output_dir}")
    tokenized_ds.save_to_disk(str(output_dir))
    print("Done!")

    return tokenized_ds


def main():
    parser = argparse.ArgumentParser(
        description="JSUT直接トークン化スクリプト"
    )
    parser.add_argument(
        "--jsut_path",
        type=str,
        default="./jsut_ver1.1",
        help="JSUTフォルダのパス (default: ./jsut_ver1.1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="出力ディレクトリ (例: ./jsut_tokenized)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lfm2",
        choices=["lfm2", "qwen3"],
        help="モデルタイプ (default: lfm2)",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=None,
        help="処理するサブセット (default: 全サブセット)",
    )

    args = parser.parse_args()

    process_jsut_direct(
        jsut_path=args.jsut_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        subsets=args.subsets,
    )


if __name__ == "__main__":
    main()
