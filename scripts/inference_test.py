import torch
from snac import SNAC
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import time
import soundfile as sf


def load_config(config_path: str):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    print("Loading configuration...")
    config = load_config("vyvotts/configs/inference/lfm2.yaml")

    # Token constants from config
    START_OF_HUMAN = config['START_OF_HUMAN']
    END_OF_TEXT = config['END_OF_TEXT']
    END_OF_HUMAN = config['END_OF_HUMAN']
    START_OF_SPEECH = config['START_OF_SPEECH']
    END_OF_SPEECH = config['END_OF_SPEECH']
    PAD_TOKEN = config['PAD_TOKEN']
    AUDIO_TOKENS_START = config['AUDIO_TOKENS_START']

    device = "cuda"
    model_name = "Vyvo/VyvoTTS-LFM2-Neuvillette"

    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    snac_model = snac_model.to(device)

    print("Loading LLM model with Flash Attention 2...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Input text
    text = "Hello, this is a test of the VyvoTTS speech synthesis system."
    print(f"Generating speech for: {text}")

    # Preprocess
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    start_token = torch.tensor([[START_OF_HUMAN]], dtype=torch.int64)
    end_tokens = torch.tensor([[END_OF_TEXT, END_OF_HUMAN]], dtype=torch.int64)
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1).to(device)
    attention_mask = torch.ones_like(modified_input_ids)

    # Generate
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=modified_input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=END_OF_SPEECH,
        )

    torch.cuda.synchronize()
    generation_time = time.time() - start_time

    # Parse audio tokens
    token_indices = (generated_ids == START_OF_SPEECH).nonzero(as_tuple=True)
    if len(token_indices[1]) > 0:
        last_idx = token_indices[1][-1].item()
        cropped = generated_ids[:, last_idx+1:]
    else:
        cropped = generated_ids

    row = cropped[0]
    row = row[row != END_OF_SPEECH]
    row_length = row.size(0)
    new_length = (row_length // 7) * 7
    trimmed = row[:new_length]
    code_list = [t.item() - AUDIO_TOKENS_START for t in trimmed]

    # Redistribute codes to SNAC layers
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(code_list)+1)//7):
        layer_1.append(code_list[7*i])
        layer_2.append(code_list[7*i+1]-4096)
        layer_3.append(code_list[7*i+2]-(2*4096))
        layer_3.append(code_list[7*i+3]-(3*4096))
        layer_2.append(code_list[7*i+4]-(4*4096))
        layer_3.append(code_list[7*i+5]-(5*4096))
        layer_3.append(code_list[7*i+6]-(6*4096))

    codes = [
        torch.tensor(layer_1).unsqueeze(0).to(device),
        torch.tensor(layer_2).unsqueeze(0).to(device),
        torch.tensor(layer_3).unsqueeze(0).to(device)
    ]

    # Decode audio
    audio = snac_model.decode(codes)
    audio_numpy = audio.detach().squeeze().cpu().numpy()

    # Save
    output_path = "test_output.wav"
    sf.write(output_path, audio_numpy, 24000)

    print(f"Audio shape: {audio.shape}")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
