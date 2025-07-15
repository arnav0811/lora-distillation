# Processing the gsm8k dataset 
import requests
import re
import random
from typing import List, Dict, Tuple
import json
import os


class GSM8KProcessor:
    def __init__(self, max_len: int = 512):
        self.max_len = max_len
        self.train_data = None
        self.test_data = None
    
    def download_dataset(self):
        train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
        test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

        def download(url):
            response = requests.get(url)
            response.raise_for_status()
            data = []
            for line in response.text.strip().split("\n"):
                if line.strip():
                    data.append(json.loads(line))
            return data
        try:
            self.train_data = download(train_url)
            self.test_data = download(test_url)
            print("Dataset downloaded")
            return True
        except Exception as e:
            print("Download Failed: {e}")
            return False
        
    def load_dataset(self):
        return self.download_dataset()
    
    def extract_answer(self, text):
        patterns = [r"#### (-?\d+\.?\d*)", r"the answer is (-?\d+\.?\d*)", r"answer: (-?\d+\.?\d*)"]
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)
        numbers = re.findall(r"-?\d+\.?\d*", text)
        return numbers[-1] if numbers else "0"
    
    def format(self, example):
        question = example["question"]
        solution = example["answer"]
        prompt = f"""<s>[INST] Solve this math problem step by step:
        {question}
        Show your work and reasoning. [/INST]
        {solution}</s>"""
        return {'text': prompt, 'question': question, 'solution': solution, 'answer': self.extract_answer(solution)}
    
    def create_training_data(self):
        if self.train_data is None or self.test_data is None:
            if not self.download_dataset():
                raise RuntimeError("Failed to download dataset")
        train_size = len(self.train_data)
        eval_size = len(self.test_data)

        train_data = random.sample(self.train_data, train_size)
        eval_data = random.sample(self.test_data, eval_size)
        
        train_data_format = [self.format(data) for data in train_data]
        eval_data_format = [self.format(data) for data in eval_data]
        
        return train_data_format, eval_data_format
    
    def save_data(self, train_data, eval_data, output_directory = "data/processed"):
        os.makedirs(output_directory, exist_ok=True)
        train_path = f"{output_directory}/train.json"
        eval_path = f"{output_directory}/eval.json"

        with open(train_path, "w") as f:
            json.dump(train_data, f, indent = 2)
        
        with open(eval_path, "w") as f:
            json.dump(eval_data, f, indent = 2)
        
        print(f"Saved training data to {train_path}")
        print(f"Saved evaluation data to {eval_path}")

        return train_path, eval_path
    
def main():
    random.seed(42)
    processor = GSM8KProcessor()
    train_data, eval_data = processor.create_training_data()
    train_path, eval_path = processor.save_data(train_data, eval_data)
    return train_path, eval_path

if __name__ == "__main__":
    main()



    