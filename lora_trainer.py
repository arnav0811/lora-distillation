from config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from peft import LoraConfig, TaskType, get_peft_model
import json
from datasets import Dataset

class LoRATrainer:
    def __init__(self):
        self.config = Config()
        self.tokenizer = None
        self.model = None
        self.train_data = None
        self.eval_data = None

    def setup_model_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.config.quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quantization_config = None
        
        base_model = AutoModelForCausalLM.from_pretrained(self.config.base_model, torch_dtype = torch.bfloat16, device_map = "auto", quantization_config = quantization_config, trust_remote_code = True)

        if self.config.checkpointing:
            base_model.gradient_checkpointing_enable()
        
        lora_config = LoraConfig(r = self.config.lora_rank, lora_alpha = self.config.lora_alpha, target_modules = self.config.lora_target_modules, lora_dropout = self.config.lora_dropout, bias = "none", task_type = TaskType.CAUSAL_LM)
        
        self.model = get_peft_model(base_model, lora_config)

        self.model.train()

        # Number of parameters that will be updated during training requires_grad
        trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in self.model.parameters())
        print("Trainable Parameters:", trainable_parameters)
        print("Total Parameters:", total_parameters)

    def load_data(self):
        with open(self.config.train_path, "r") as f:
            self.train_data = json.load(f)
        with open(self.config.eval_path, "r") as f:
            self.eval_data = json.load(f)

        train_text = [i['text'] for i in self.train_data]
        eval_text = [i['text'] for i in self.eval_data]
        
        def tokenize_data(text):
            return self.tokenizer(text['text'], truncation = True, padding = False, max_length = self.config.max_len, return_tensors = None)

        self.train_data = Dataset.from_dict({"text": train_text})
        self.eval_data = Dataset.from_dict({"text": eval_text})

        self.train_data = self.train_data.map(tokenize_data, batched = True, remove_columns = ["text"])
        self.eval_data = self.eval_data.map(tokenize_data, batched = True, remove_columns = ["text"])
        print("Tokenized")

    
    def train(self):
        training_args = TrainingArguments(
            output_dir = self.config.output_directory,
            num_train_epochs = self.config.epochs,
            per_device_train_batch_size = self.config.batch_size,
            per_device_eval_batch_size = self.config.batch_size,
            gradient_accumulation_steps = self.config.gradient_accumulation_steps,
            warmup_steps = self.config.warmup_steps,
            learning_rate = self.config.lr,
            bf16 = True,
            logging_steps = self.config.logging_steps,
            eval_strategy = "steps",
            eval_steps = self.config.eval_steps,
            save_steps = self.config.save_steps,
            save_strategy = "steps",
            gradient_checkpointing = False,
            report_to = "none"
        )

        # Intelligent Batching without masked lang modelling
        data_collator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, mlm = False)
        trainer = Trainer(model = self.model, args = training_args, train_dataset = self.train_data, eval_dataset = self.eval_data, data_collator = data_collator)
        trainer.train()

        trainer.save_model(self.config.output_directory)
        self.tokenizer.save_pretrained(self.config.output_directory)
        return self.config.output_directory

def main():
    trainer = LoRATrainer()
    trainer.setup_model_tokenizer()
    trainer.load_data()
    trainer.train()

if __name__ == "__main__":
    main()
