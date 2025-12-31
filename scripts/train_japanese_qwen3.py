"""
日本語ファインチューニングスクリプト (Qwen3版)

Qwen3モデルを使用した日本語TTSファインチューニング。
Qwen3はvocab size 151,669で日本語トークン効率が向上。

使用方法:
    uv run python scripts/train_japanese_qwen3.py
    uv run python scripts/train_japanese_qwen3.py --dataset_path ./moe_tokenized_qwen3 --epochs 3
"""

from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import torch
import wandb
import argparse

# Qwen3 0.6B用デフォルト設定
DEFAULT_CONFIG = {
    "dataset_path": "./moe_tokenized_qwen3",
    "model_name": "Vyvo/Qwen3-0.6B-PT",  # 0.6Bプリトレーニングモデル
    "epochs": 3,
    "batch_size": 8,                # 0.6Bなのでbatch 8可能
    "learning_rate": 1e-5,          # より保守的な学習率
    "save_steps": 500,
    "warmup_steps": 200,
    "gradient_accumulation": 4,     # 実効バッチサイズ32 (8*4)
    "pad_token": 151676,            # Qwen3のPAD_TOKEN
    "save_folder": "checkpoints-qwen3-japanese",
    "project_name": "vyvotts-japanese-qwen3",
    "run_name": "moe-qwen3-0.6b-finetune",
}


def main():
    parser = argparse.ArgumentParser(description="日本語ファインチューニング (Qwen3)")
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_CONFIG["dataset_path"],
                        help="トークン化済みデータセットのパス")
    parser.add_argument("--model_name", type=str, default=DEFAULT_CONFIG["model_name"],
                        help="ベースモデル名")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"],
                        help="エポック数")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                        help="バッチサイズ")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="学習率")
    parser.add_argument("--save_steps", type=int, default=DEFAULT_CONFIG["save_steps"],
                        help="保存間隔（ステップ数）")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"],
                        help="ウォームアップステップ数")
    parser.add_argument("--gradient_accumulation", type=int, default=DEFAULT_CONFIG["gradient_accumulation"],
                        help="勾配蓄積ステップ数")
    parser.add_argument("--save_folder", type=str, default=DEFAULT_CONFIG["save_folder"],
                        help="チェックポイント保存先")
    parser.add_argument("--no_wandb", action="store_true",
                        help="WandBを無効化")

    args = parser.parse_args()

    print("=" * 50)
    print("日本語ファインチューニング (Qwen3)")
    print("=" * 50)
    print(f"Dataset: {args.dataset_path}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Save folder: {args.save_folder}")
    print("=" * 50)

    # データセット読み込み（Arrow/Parquet自動検出）
    print("\nLoading dataset...")
    dataset_path = Path(args.dataset_path)
    parquet_file = dataset_path / "data.parquet"

    if parquet_file.exists():
        # Parquet形式（高速読み込み）
        print(f"  Detected Parquet format: {parquet_file}")
        ds = Dataset.from_parquet(str(parquet_file))
    else:
        # Arrow形式（従来）
        print(f"  Detected Arrow format: {dataset_path}")
        ds = load_from_disk(args.dataset_path)

    print(f"Dataset loaded: {len(ds)} samples")

    # モデル読み込み
    print("\nLoading model...")
    # トークナイザーはベースモデルから読み込む
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # モデルを読み込む
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"Model loaded: {args.model_name}")

    # WandB初期化
    if not args.no_wandb:
        wandb.init(
            project=DEFAULT_CONFIG["project_name"],
            name=DEFAULT_CONFIG["run_name"]
        )

    # トレーニング設定（最適化済み）
    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=50,  # 最適化: 10 -> 50
        bf16=True,
        tf32=True,  # TF32行列演算有効化
        output_dir=f"./{args.save_folder}",
        report_to="wandb" if not args.no_wandb else "none",
        save_steps=args.save_steps,
        save_total_limit=3,  # チェックポイント数制限
        remove_unused_columns=True,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # PyTorch 2.0+最適化
        optim="adamw_torch_fused",
        max_grad_norm=1.0,  # 勾配クリッピング
        weight_decay=0.01,  # 正則化
        # DataLoader最適化
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        dataloader_drop_last=True,
    )

    # データコレーター（最適化版: pad_sequenceを使用）
    def data_collator(features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        # torch.nn.utils.rnn.pad_sequenceを使用した高速パディング
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=DEFAULT_CONFIG["pad_token"]
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels,
        }

    # トレーナー
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    # トレーニング開始
    print("\nStarting training...")
    trainer.train()

    # 最終モデル保存
    print(f"\nSaving final model to {args.save_folder}...")
    trainer.save_model(args.save_folder)
    tokenizer.save_pretrained(args.save_folder)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
