import torch
from torch.utils.data import Dataset

class CustomQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answers']['text'][0]

        inputs = self.tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        start_positions = torch.tensor([i for i, id in enumerate(input_ids.tolist()) if id == answer_tokens[0]])
        end_positions = torch.tensor([i for i, id in enumerate(input_ids.tolist()) if id == answer_tokens[-1]])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions
        }
