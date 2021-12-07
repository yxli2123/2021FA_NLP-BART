from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch


class BaseDataset(Dataset):
    def __init__(self, dataset_name, model_name, split, seq_len):
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.seq_len = seq_len
        self.split = split

        self.data = None
        self.preprocess_dataset()

    def get_tokenizer(self):
        return self.tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        if self.dataset_name == 'multi_nli':
            input_encoded = self.tokenizer(text=record['text'],
                                           add_special_tokens=False,
                                           padding='max_length',
                                           max_length=self.seq_len,
                                           truncation=True,
                                           return_attention_mask=True)

            input_token_ids = torch.tensor(input_encoded['input_ids'])
            input_attn_mask = torch.tensor(input_encoded['attention_mask'])

            sample = {'input_text': record['text'],
                      'input_token_ids': input_token_ids,
                      'input_attn_mask': input_attn_mask,
                      'label': record['label']}

        elif self.dataset_name == 'cnn_dailymail':
            input_encoded = self.tokenizer.encode_plus(text=record['text'],
                                                       add_special_tokens=False,
                                                       padding='max_length',
                                                       max_length=self.seq_len,
                                                       truncation=True,
                                                       return_attention_mask=True)
            target_encoded = self.tokenizer.encode_plus(text=record['label'],
                                                        add_special_tokens=False,
                                                        padding='max_length',
                                                        max_length=self.seq_len // 8,
                                                        truncation=True,
                                                        return_attention_mask=True)

            input_token_ids = torch.tensor(input_encoded['input_ids'])
            input_attn_mask = torch.tensor(input_encoded['attention_mask'])

            target_token_ids = torch.tensor(target_encoded['input_ids'])
            target_attn_mask = torch.tensor(target_encoded['attention_mask'])

            # Output
            sample = {'input_text': record['text'],
                      'label_text': record['label'],
                      'input_token_ids': input_token_ids,
                      'input_attn_mask': input_attn_mask,
                      'label': target_token_ids}
        else:
            raise KeyError("expected multi_nli or cnn_dailymail dataset, but got other dataset name")

        return sample

    def preprocess_dataset(self):
        self.data = []
        if self.dataset_name == 'multi_nli':
            dataset = load_dataset("multi_nli")
            split = 'train' if self.split == 'train' else 'validation_matched'
            data = dataset[split]
            for row in data:
                self.data.append({'text': f"{row['premise']}\n{row['hypothesis']}",
                                  'label': int(row['label'])})

        elif self.dataset_name == 'cnn_dailymail':
            dataset = load_dataset("cnn_dailymail", "3.0.0")
            split = 'train' if self.split == 'train' else 'validation'
            data = dataset['data'][split]
            for row in data:
                self.data.append({'text': row['article'],
                                  'label': row['highlights']})

        else:
            raise KeyError("expected multi_nli or cnn_dailymail dataset, but got other dataset name")


if __name__ == '__main__':
    dataset_ = load_dataset("cnn_dailymail", "3.0.0")
    print("Done!")

