"""
日本語ファインチューニングスクリプト

ローカルのトークン化済みデータセットを使用してファインチューニングを実行します。

使用方法:
    uv run python scripts/train_japanese.py
"""

from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, AutoTokenizer
from transformers.models.lfm2 import Lfm2ForCausalLM
import torch
import wandb
import argparse

# デフォルト設定
DEFAULT_CONFIG = {
    "dataset_path": "./jsut_tokenized",
    "model_name": "Vyvo/VyvoTTS-LFM2-Neuvillette",
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 5e-5,
    "save_steps": 1000,
    "pad_token": 64407,
    "save_folder": "checkpoints-japanese",
    "project_name": "vyvotts-japanese",
    "run_name": "jsut-finetune",
}


def main():
    parser = argparse.ArgumentParser(description="日本語ファインチューニング")
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
    parser.add_argument("--save_folder", type=str, default=DEFAULT_CONFIG["save_folder"],
                        help="チェックポイント保存先")
    parser.add_argument("--no_wandb", action="store_true",
                        help="WandBを無効化")

    args = parser.parse_args()

    print("=" * 50)
    print("日本語ファインチューニング")
    print("=" * 50)
    print(f"Dataset: {args.dataset_path}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save folder: {args.save_folder}")
    print("=" * 50)

    # データセット読み込み
    print("\nLoading dataset...")
    ds = load_from_disk(args.dataset_path)
    print(f"Dataset loaded: {len(ds)} samples")

    # モデル読み込み
    print("\nLoading model...")
    # トークナイザーはベースモデルから読み込む
    tokenizer = AutoTokenizer.from_pretrained("LiquidAI/LFM2-350M")
    # モデルを読み込む
    model = Lfm2ForCausalLM.from_pretrained(
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

    # トレーニング設定
    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=10,
        bf16=True,
        output_dir=f"./{args.save_folder}",
        report_to="wandb" if not args.no_wandb else "none",
        save_steps=args.save_steps,
        remove_unused_columns=True,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
    )

    # データコレーター
    def data_collator(features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # パディング
        max_len = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lab in zip(input_ids, attention_mask, labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [DEFAULT_CONFIG["pad_token"]] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
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
