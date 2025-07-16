import os

class Config:
    base_model = "mistralai/Mistral-7B-Instruct-v0.1"
    lora_rank = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

    max_len = 512
    batch_size = 2
    gradient_accumulation_steps = 4
    lr = 2e-4
    epochs = 3
    warmup_steps = 100

    quantization = True
    checkpointing = True

    train_path = "data/processed/train.json"
    eval_path = "data/processed/eval.json"

    output_directory = "models/lora_mistral_gsm8k"

    logging_steps = 10
    eval_steps = 100
    save_steps = 200

def setup_directory():
    os.makedirs("models")
    os.makedirs(LoRAConfig.output_directory)

if __name__ == "__main__":
    setup_directory()
    config = LoRAConfig()
    print("Base Model: ", config.base_model)
    print("LORA Rank: ", config.lora_rank)



    