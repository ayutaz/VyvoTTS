"""
日本語TTSモデル推論テストスクリプト (Qwen3版)

Qwen3ベースの日本語モデルで音声生成テストを実行します。

使用方法:
    uv run python scripts/test_japanese_qwen3.py
    uv run python scripts/test_japanese_qwen3.py --model ./checkpoints-qwen3-japanese --temperature 0.5
"""

from snac import SNAC
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Optional, Dict, Any
import yaml
import time
import argparse
import re
import unicodedata


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def normalize_japanese_text(text: str) -> str:
    """
    日本語テキストを正規化する。
    """
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class Qwen3JapaneseInference:
    """日本語TTS推論エンジン (Qwen3版)"""

    def __init__(self, model_name: str = "./checkpoints-qwen3-japanese", device: str = "cuda"):
        # Load Qwen3 configuration
        self.config = load_config("vyvotts/configs/inference/qwen3.yaml")

        # Set token constants from config
        self.TOKENIZER_LENGTH = self.config['TOKENIZER_LENGTH']
        self.START_OF_TEXT = self.config['START_OF_TEXT']
        self.END_OF_TEXT = self.config['END_OF_TEXT']
        self.START_OF_SPEECH = self.config['START_OF_SPEECH']
        self.END_OF_SPEECH = self.config['END_OF_SPEECH']
        self.START_OF_HUMAN = self.config['START_OF_HUMAN']
        self.END_OF_HUMAN = self.config['END_OF_HUMAN']
        self.START_OF_AI = self.config['START_OF_AI']
        self.END_OF_AI = self.config['END_OF_AI']
        self.PAD_TOKEN = self.config['PAD_TOKEN']
        self.AUDIO_TOKENS_START = self.config['AUDIO_TOKENS_START']

        self.device = device

        # SNAC model
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to(self.device)

        # LLM model with flash_attention_2
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        # Use base Qwen3 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def _preprocess_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        all_input_ids = []
        for prompt in prompts:
            # Normalize Japanese text
            prompt = normalize_japanese_text(prompt)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            all_input_ids.append(input_ids)

        start_token = torch.tensor([[self.START_OF_HUMAN]], dtype=torch.int64)
        end_tokens = torch.tensor([[self.END_OF_TEXT, self.END_OF_HUMAN]], dtype=torch.int64)

        all_modified_input_ids = []
        for input_ids in all_input_ids:
            modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)
            all_modified_input_ids.append(modified_input_ids)

        all_padded_tensors = []
        all_attention_masks = []
        max_length = max([m.shape[1] for m in all_modified_input_ids])

        for modified_input_ids in all_modified_input_ids:
            padding = max_length - modified_input_ids.shape[1]
            padded_tensor = torch.cat([torch.full((1, padding), self.PAD_TOKEN, dtype=torch.int64), modified_input_ids], dim=1)
            attention_mask = torch.cat([torch.zeros((1, padding), dtype=torch.int64), torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
            all_padded_tensors.append(padded_tensor)
            all_attention_masks.append(attention_mask)

        all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        return all_padded_tensors.to(self.device), all_attention_masks.to(self.device)

    def _redistribute_codes(self, code_list: List[int]) -> torch.Tensor:
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))

        codes = [torch.tensor(layer_1).unsqueeze(0).to(self.device),
                 torch.tensor(layer_2).unsqueeze(0).to(self.device),
                 torch.tensor(layer_3).unsqueeze(0).to(self.device)]
        return self.snac_model.decode(codes)

    def generate(
        self,
        text: str,
        output_path: Optional[str] = None,
        temperature: float = 0.5,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 1200,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        torch.cuda.synchronize()
        total_start = time.time()

        # Preprocess
        preprocess_start = time.time()
        input_ids, attention_mask = self._preprocess_prompts([text])
        preprocess_time = time.time() - preprocess_start

        # Generate
        gen_start = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.END_OF_SPEECH,
            )
        torch.cuda.synchronize()
        generation_time = time.time() - gen_start

        # Parse audio
        audio_start = time.time()
        token_indices = (generated_ids == self.START_OF_SPEECH).nonzero(as_tuple=True)
        if len(token_indices[1]) > 0:
            last_idx = token_indices[1][-1].item()
            cropped = generated_ids[:, last_idx+1:]
        else:
            cropped = generated_ids

        processed = []
        for row in cropped:
            masked = row[row != self.END_OF_SPEECH]
            processed.append(masked)

        code_lists = []
        for row in processed:
            length = row.size(0)
            new_len = (length // 7) * 7
            trimmed = row[:new_len]
            trimmed = [t - self.AUDIO_TOKENS_START for t in trimmed]
            code_lists.append(trimmed)

        samples = []
        for codes in code_lists:
            if len(codes) >= 7:
                audio = self._redistribute_codes(codes)
                samples.append(audio)

        torch.cuda.synchronize()
        audio_time = time.time() - audio_start

        timing = {
            'preprocessing_time': preprocess_time,
            'generation_time': generation_time,
            'audio_processing_time': audio_time,
            'total_time': time.time() - total_start
        }

        audio = samples[0] if samples else None

        if output_path and audio is not None:
            import soundfile as sf
            audio_np = audio.detach().squeeze().cpu().numpy()
            sf.write(output_path, audio_np, 24000)

        return audio, timing


def main():
    parser = argparse.ArgumentParser(description="日本語TTS推論テスト (Qwen3)")
    parser.add_argument("--model", type=str, default="./checkpoints-qwen3-japanese",
                        help="モデルのパス")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="サンプリング温度（低いほど安定、デフォルト: 0.5）")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-pサンプリング（デフォルト: 0.9）")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="繰り返しペナルティ（デフォルト: 1.2）")
    parser.add_argument("--output_prefix", type=str, default="output_qwen3_jp",
                        help="出力ファイル名のプレフィックス")
    parser.add_argument("--text", type=str, default=None,
                        help="生成するテキスト（指定しない場合はデフォルトテキストを使用）")

    args = parser.parse_args()

    print("=" * 50)
    print("日本語TTSモデル推論テスト (Qwen3)")
    print("=" * 50)
    print(f"モデル: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Repetition Penalty: {args.repetition_penalty}")
    print("=" * 50)

    # ローカルチェックポイントを使用
    print("\nモデルを読み込み中...")
    engine = Qwen3JapaneseInference(
        model_name=args.model,
        device="cuda"
    )
    print("モデルの読み込み完了")

    # 日本語テキストで音声生成
    if args.text:
        test_texts = [args.text]
    else:
        test_texts = [
            "こんにちは、私は日本語の音声合成モデルです。",
            "今日はいい天気ですね。",
            "音声合成技術は、テキストから自然な音声を生成します。",
        ]

    print(f"\n{len(test_texts)}個のテキストで音声生成テストを開始...")
    print("-" * 50)

    for i, text in enumerate(test_texts):
        print(f"\n[{i+1}/{len(test_texts)}] テキスト: {text}")
        output_path = f"{args.output_prefix}_{i}.wav"

        try:
            audio, timing = engine.generate(
                text,
                output_path=output_path,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            if audio is not None:
                print(f"  生成成功: {output_path}")
                print(f"  音声長: {audio.shape[-1] / 24000:.2f}秒")
                print(f"  処理時間: {timing['total_time']:.2f}秒")
                print(f"    - 前処理: {timing['preprocessing_time']:.3f}秒")
                print(f"    - 生成: {timing['generation_time']:.3f}秒")
                print(f"    - 音声変換: {timing['audio_processing_time']:.3f}秒")
            else:
                print(f"  生成失敗: 音声データなし")
        except Exception as e:
            print(f"  エラー: {e}")

    print("\n" + "=" * 50)
    print("テスト完了")
    print("=" * 50)


if __name__ == "__main__":
    main()
