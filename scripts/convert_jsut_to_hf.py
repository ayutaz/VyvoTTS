"""
JSUT → HuggingFace Dataset 変換スクリプト

JSUTデータセットをHuggingFace Dataset形式に変換します。

使用方法:
    # ローカル保存
    uv run python scripts/convert_jsut_to_hf.py --output_dir ./jsut_dataset

    # HuggingFace Hubにアップロード
    uv run python scripts/convert_jsut_to_hf.py --push_to_hub username/jsut-japanese-tts
"""

import argparse
import os
from pathlib import Path
from datasets import Dataset, Audio
from tqdm import tqdm


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


def load_transcript(transcript_path: Path) -> dict:
    """
    トランスクリプトファイルを読み込む

    Args:
        transcript_path: transcript_utf8.txt のパス

    Returns:
        {ファイル名: テキスト} の辞書
    """
    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # コロン区切り: BASIC5000_0001:テキスト
            parts = line.split(":", 1)
            if len(parts) == 2:
                filename, text = parts
                transcripts[filename] = text
    return transcripts


def process_jsut_folder(jsut_path: Path, subsets: list = None) -> list:
    """
    JSUTフォルダを処理してデータリストを作成

    Args:
        jsut_path: JSUTのルートフォルダパス
        subsets: 処理するサブセットのリスト（Noneの場合は全サブセット）

    Returns:
        [{audio: path, text: str, subset: str}, ...] のリスト
    """
    if subsets is None:
        subsets = JSUT_SUBSETS

    data = []

    for subset in subsets:
        subset_path = jsut_path / subset
        if not subset_path.exists():
            print(f"Warning: {subset} not found, skipping...")
            continue

        transcript_path = subset_path / "transcript_utf8.txt"
        wav_dir = subset_path / "wav"

        if not transcript_path.exists() or not wav_dir.exists():
            print(f"Warning: {subset} missing transcript or wav folder, skipping...")
            continue

        # トランスクリプト読み込み
        transcripts = load_transcript(transcript_path)

        # WAVファイルを処理
        for filename, text in tqdm(transcripts.items(), desc=f"Processing {subset}"):
            wav_path = wav_dir / f"{filename}.wav"
            if wav_path.exists():
                data.append({
                    "audio": str(wav_path.absolute()),
                    "text": text,
                    "subset": subset,
                })
            else:
                print(f"Warning: {wav_path} not found, skipping...")

    return data


def create_dataset(data: list, sample_rate: int = 24000) -> Dataset:
    """
    データリストからHuggingFace Datasetを作成

    Args:
        data: [{audio: path, text: str, subset: str}, ...] のリスト
        sample_rate: 目標サンプルレート

    Returns:
        HuggingFace Dataset
    """
    dataset = Dataset.from_list(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="JSUT → HuggingFace Dataset 変換スクリプト"
    )
    parser.add_argument(
        "--jsut_path",
        type=str,
        default="./jsut_ver1.1",
        help="JSUTデータセットのパス (default: ./jsut_ver1.1)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="ローカル保存先ディレクトリ",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hubにアップロードする場合のリポジトリ名 (例: username/jsut-japanese-tts)",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        nargs="+",
        default=None,
        help="処理するサブセット (default: 全サブセット)",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=24000,
        help="目標サンプルレート (default: 24000)",
    )

    args = parser.parse_args()

    # 引数チェック
    if args.output_dir is None and args.push_to_hub is None:
        parser.error("--output_dir または --push_to_hub のいずれかを指定してください")

    jsut_path = Path(args.jsut_path)
    if not jsut_path.exists():
        parser.error(f"JSUTフォルダが見つかりません: {jsut_path}")

    print(f"JSUT path: {jsut_path}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Subsets: {args.subsets or 'all'}")
    print()

    # データ処理
    print("Loading JSUT dataset...")
    data = process_jsut_folder(jsut_path, args.subsets)
    print(f"Total samples: {len(data)}")
    print()

    # Dataset作成
    print("Creating HuggingFace Dataset...")
    dataset = create_dataset(data, args.sample_rate)
    print(f"Dataset created: {dataset}")
    print()

    # 保存
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path}...")
        dataset.save_to_disk(str(output_path))
        print("Done!")

    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}...")
        dataset.push_to_hub(args.push_to_hub)
        print("Done!")


if __name__ == "__main__":
    main()
