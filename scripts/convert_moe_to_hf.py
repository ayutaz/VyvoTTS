"""
MOE Dataset → HuggingFace Dataset 変換スクリプト

MOEデータセット（マルチスピーカー）をHuggingFace Dataset形式に変換します。

使用方法:
    uv run python scripts/convert_moe_to_hf.py --moe_path D:/moe_top20 --output_dir ./moe_dataset_hf
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset, Audio
from tqdm import tqdm


def process_moe_folder(moe_path: Path, text_field: str = "anime_whisper_transcription") -> list:
    """
    MOEフォルダを処理してデータリストを作成

    Args:
        moe_path: MOEデータセットのルートフォルダパス
        text_field: JSONから抽出するテキストフィールド

    Returns:
        [{audio: path, text: str, source: str}, ...] のリスト
    """
    data = []

    # 話者ディレクトリを取得（calc_total.ps1などのファイルを除外）
    speaker_dirs = [d for d in moe_path.iterdir() if d.is_dir()]
    print(f"Found {len(speaker_dirs)} speaker directories")

    for speaker_dir in tqdm(speaker_dirs, desc="Processing speakers"):
        speaker_id = speaker_dir.name

        # MOEの構造: {speaker_id}/data/{speaker_id}/wav/
        wav_dir = speaker_dir / "data" / speaker_id / "wav"

        if not wav_dir.exists():
            print(f"Warning: {wav_dir} not found, skipping speaker {speaker_id}...")
            continue

        # JSONファイルを探索
        json_files = list(wav_dir.glob("*.json"))

        for json_path in json_files:
            # 対応するWAVファイルのパス
            wav_path = json_path.with_suffix(".wav")

            if not wav_path.exists():
                continue

            try:
                # JSONを読み込み
                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # テキストを取得
                text = metadata.get(text_field)
                if not text:
                    # フォールバック: 別のフィールドを試す
                    text = metadata.get("parakeet_jp_transcription")

                if not text:
                    print(f"Warning: No text found in {json_path}, skipping...")
                    continue

                data.append({
                    "audio": str(wav_path.absolute()),
                    "text": text,
                    "source": speaker_id,  # マルチスピーカー用
                })

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse {json_path}: {e}")
                continue

    return data


def create_dataset(data: list, sample_rate: int = 24000) -> Dataset:
    """
    データリストからHuggingFace Datasetを作成

    Args:
        data: [{audio: path, text: str, source: str}, ...] のリスト
        sample_rate: 目標サンプルレート

    Returns:
        HuggingFace Dataset
    """
    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="MOE Dataset → HuggingFace Dataset 変換スクリプト"
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
        default="./moe_dataset_hf",
        help="ローカル保存先ディレクトリ (default: ./moe_dataset_hf)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="anime_whisper_transcription",
        choices=["anime_whisper_transcription", "parakeet_jp_transcription"],
        help="使用するテキストフィールド (default: anime_whisper_transcription)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="目標サンプルレート (default: 24000)",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hubにアップロードする場合のリポジトリ名",
    )

    args = parser.parse_args()

    moe_path = Path(args.moe_path)
    if not moe_path.exists():
        parser.error(f"MOEフォルダが見つかりません: {moe_path}")

    print("=" * 60)
    print("MOE Dataset → HuggingFace Dataset 変換")
    print("=" * 60)
    print(f"MOE path: {moe_path}")
    print(f"Output dir: {args.output_dir}")
    print(f"Text field: {args.text_field}")
    print(f"Sample rate: {args.sample_rate}")
    print("=" * 60)
    print()

    # データ処理
    print("Loading MOE dataset...")
    data = process_moe_folder(moe_path, args.text_field)
    print(f"\nTotal samples: {len(data)}")

    # 話者ごとの統計
    speaker_counts = {}
    for item in data:
        speaker = item["source"]
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

    print(f"Speakers: {len(speaker_counts)}")
    print("\nSamples per speaker:")
    for speaker, count in sorted(speaker_counts.items(), key=lambda x: -x[1]):
        print(f"  {speaker}: {count}")
    print()

    # Dataset作成
    print("Creating HuggingFace Dataset...")
    dataset = create_dataset(data, args.sample_rate)
    print(f"Dataset created: {dataset}")
    print()

    # 保存
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path}...")
    dataset.save_to_disk(str(output_path))
    print("Done!")

    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}...")
        dataset.push_to_hub(args.push_to_hub)
        print("Done!")


if __name__ == "__main__":
    main()
