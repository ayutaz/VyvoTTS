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

# デフォルト設定（v3: MOE 53Kデータセット用に最適化）
DEFAULT_CONFIG = {
    "dataset_path": "./moe_tokenized_lfm2",
    "model_name": "Vyvo/VyvoTTS-LFM2-Neuvillette",
    "epochs": 3,                    # 大規模データなので3エポック
    "batch_size": 16,               # RTX 4090で16可能
    "learning_rate": 2e-5,          # 5e-5 → 2e-5（プリトレーニング済みモデルなので控えめに）
    "save_steps": 1000,             # 500 → 1000（大規模データ用）
    "warmup_steps": 300,            # 500 → 300（大規模データは早めに本学習へ）
    "gradient_accumulation": 4,     # 2 → 4（実効バッチサイズ64で安定化）
    "pad_token": 64407,
    "save_folder": "checkpoints-lfm2-japanese",
    "project_name": "vyvotts-japanese-lfm2",
    "run_name": "moe-53k-lfm2-finetune",
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
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"],
                        help="ウォームアップステップ数")
    parser.add_argument("--gradient_accumulation", type=int, default=DEFAULT_CONFIG["gradient_accumulation"],
                        help="勾配蓄積ステップ数")
    parser.add_argument("--save_folder", type=str, default=DEFAULT_CONFIG["save_folder"],
                        help="チェックポイント保存先")
    parser.add_argument("--no_wandb", action="store_true",
                        help="WandBを無効化")
    parser.add_argument("--no_compile", action="store_true",
                        help="torch.compileを無効化")

    args = parser.parse_args()

    print("=" * 50)
    print("日本語ファインチューニング (v3: LFM2 MOE 53K)")
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

    # torch.compileで最適化（PyTorch 2.0+）
    if not args.no_compile:
        try:
            model = torch.compile(model)
            print("torch.compile applied successfully")
        except Exception as e:
            print(f"torch.compile skipped: {e}")

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
        logging_steps=50,  # 最適化: 10 -> 50 (ログ頻度削減)
        bf16=True,
        output_dir=f"./{args.save_folder}",
        report_to="wandb" if not args.no_wandb else "none",
        save_steps=args.save_steps,
        remove_unused_columns=True,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        # DataLoader最適化フラグ
        dataloader_num_workers=4,  # データローディング並列化
        dataloader_pin_memory=True,  # CPU→GPU転送高速化
        dataloader_prefetch_factor=2,  # プリフェッチ有効化
        dataloader_drop_last=True,  # 小さいバッチを避ける
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
