from torch.utils.data import Dataset
# import datasets
from datasets import load_dataset 
import transformers
import torch
from typing import List, Dict, Any


class SFTDataset(Dataset):
    def __init__(self, data_path , tokenizer, model_type, max_len):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = tokenizer
        self.max_len = max_len

        if 'llama' in model_type.lower():
            print(f'llama3 dataset')
            self.system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
            self.user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            self.assistant_format='{content}<|eot_id|>\n'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        input_ids, target_mask = [], []

        # human = data['rag_input']
        # assistant = data['output']

        human = self.user_format.format(content=data['rag_input'])
        assistant = self.assistant_format.format(content=data['output'])

        input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
        output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

        input_ids += input_tokens + output_tokens
        target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)
        input_ids = input_ids[:self.max_len]
        target_mask = target_mask[:self.max_len]
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': target_mask,
        }

        # import pdb; pdb.set_trace()
        return inputs
    

class SFTDataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Compute per-batch max length
        lengths = [len(x['input_ids']) for x in batch if x['input_ids'] is not None]
        # Clip to max_seq_length if exceeding
        batch_max_len = min(max(lengths), self.max_seq_length)
        # batch_max_len = self.max_seq_length

        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # truncate and padding
        for x in batch:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            # import pdb; pdb.set_trace()
            target_mask = x['labels']
            
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.max_seq_length]
            attention_mask = attention_mask[:self.max_seq_length]
            target_mask = target_mask[:self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

        # Convert lists to final model-input tensors
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': labels
        }
        return inputs
    


class InputContravasiveDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, model_type: str, model_max_length: int):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        if 'llama' in model_type.lower():
            print(f'llama3 dataset')
            self.system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
            self.user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            self.assistant_format='{content}<|eot_id|>\n'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Llama3 chat template: f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        data = self.dataset[idx]
        rag_input_ids, raw_input_ids, rag_labels, raw_labels = [], [], [], []

        rag_human  = self.user_format.format(content=data['rag_input'])
        rag_assistant = self.assistant_format.format(content=data['output'])
        raw_human = self.user_format.format(content=data['raw_input'])
        raw_assistant = self.assistant_format.format(content=data['output'])
        
        rag_input_tokens = self.tokenizer.encode(rag_human, add_special_tokens=False)
        rag_output_tokens = self.tokenizer.encode(rag_assistant, add_special_tokens=False)
        raw_input_tokens = self.tokenizer.encode(raw_human, add_special_tokens=False)
        raw_output_tokens = self.tokenizer.encode(raw_assistant, add_special_tokens=False)

        rag_input_ids = rag_input_tokens + rag_output_tokens
        rag_target_mask = [0] * len(rag_input_tokens) + [1] * len(rag_output_tokens)
        raw_input_ids = raw_input_tokens + raw_output_tokens
        raw_target_mask = [0] * len(raw_input_tokens) + [1] * len(raw_output_tokens)

        assert len(rag_input_ids) == len(rag_target_mask)
        assert len(raw_input_ids) == len(raw_target_mask)

        rag_input_ids = rag_input_ids[:self.model_max_length]
        raw_input_ids = raw_input_ids[:self.model_max_length]
        rag_target_mask = rag_target_mask[:self.model_max_length]
        raw_target_mask = raw_target_mask[:self.model_max_length]
        
        rag_attention_mask = [1] * len(rag_input_ids)
        raw_attention_mask = [1] * len(raw_input_ids)
        assert len(rag_input_ids) == len(rag_target_mask) == len(rag_attention_mask) 
        assert len(raw_input_ids) == len(raw_target_mask) == len(raw_attention_mask)
        rag_inputs = {
            'input_ids': rag_input_ids,
            'attention_mask': rag_attention_mask,
            'target_mask': rag_target_mask
        }
        raw_inputs = {
            'input_ids': raw_input_ids,
            'attention_mask': raw_attention_mask,
            'target_mask': raw_target_mask
        }
        return rag_inputs, raw_inputs

# 3. collator
class InputContravasiveCollator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, model_max_length: int):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        rag_inputs = [f[0] for f in features]
        raw_inputs = [f[1] for f in features]

        rag_max_lengths = [len(x['input_ids']) for x in rag_inputs if x['input_ids'] is not None]
        rag_batch_max_len = min(max(rag_max_lengths), self.model_max_length)

        rag_input_ids_batch, rag_attention_mask_batch, rag_target_mask_batch = [], [], []        
        for x in rag_inputs:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = rag_batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            target_mask = target_mask[:self.model_max_length]

            rag_input_ids_batch.append(input_ids)
            rag_attention_mask_batch.append(attention_mask)
            rag_target_mask_batch.append(target_mask)

        rag_input_ids_batch = torch.tensor(rag_input_ids_batch, dtype=torch.long)
        rag_attention_mask_batch = torch.tensor(rag_attention_mask_batch, dtype=torch.long)
        rag_target_mask_batch = torch.tensor(rag_target_mask_batch, dtype=torch.long)
        
        rag_labels = torch.where(rag_target_mask_batch == 1, rag_input_ids_batch, -100)
        
        # Handle raw (parametric-memory) branch
        raw_max_lengths = [len(x['input_ids']) for x in raw_inputs if x['input_ids'] is not None]
        raw_batch_max_len = min(max(raw_max_lengths), self.model_max_length)

        # truncate and pad
        raw_input_ids_batch, raw_attention_mask_batch, raw_target_mask_batch = [], [], []        
        for x in raw_inputs:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = raw_batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            target_mask = target_mask[:self.model_max_length]

            raw_input_ids_batch.append(input_ids)
            raw_attention_mask_batch.append(attention_mask)
            raw_target_mask_batch.append(target_mask)

        # Convert lists to final model-input tensors
        raw_input_ids_batch = torch.tensor(raw_input_ids_batch, dtype=torch.long)
        raw_attention_mask_batch = torch.tensor(raw_attention_mask_batch, dtype=torch.long)
        raw_target_mask_batch = torch.tensor(raw_target_mask_batch, dtype=torch.long)

        raw_labels = torch.where(raw_target_mask_batch == 1, raw_input_ids_batch, -100)
        return {
            'rag_input_ids': rag_input_ids_batch,
            'rag_attention_mask': rag_attention_mask_batch,
            'rag_labels': rag_labels,
            'raw_input_ids': raw_input_ids_batch,
            'raw_attention_mask':raw_attention_mask_batch,            
            'raw_labels': raw_labels
        }


class InputContrastiveLambdaDataset(InputContravasiveDataset):
    """APKS adaptive-lambda variant of InputContravasiveDataset.

    Identical rag/raw construction, but each item additionally carries a
    precomputed per-sample suppression coefficient read from the ``lambda``
    field of the data. This lambda is produced offline by the frozen
    base-model risk detector (see scripts/step2_annotate_lambda.py), so the
    suppression strength used during training matches the one the detector
    will assign at inference time.
    """
    def __getitem__(self, idx):
        rag_inputs, raw_inputs = super().__getitem__(idx)
        sample_lambda = float(self.dataset[idx].get('lambda', 0.0))
        return rag_inputs, raw_inputs, sample_lambda


class InputContrastiveLambdaCollator(InputContravasiveCollator):
    """Collator for InputContrastiveLambdaDataset.

    Reuses the parent padding/label logic for the rag/raw tensors and appends
    a ``sample_lambda`` tensor of shape (batch,) that the trainer applies to
    the UA-FFNs before the forward pass.
    """
    def __call__(self, features):
        base_features = [(f[0], f[1]) for f in features]
        batch = super().__call__(base_features)
        batch['sample_lambda'] = torch.tensor(
            [f[2] for f in features], dtype=torch.float32
        )
        return batch


class BothContravasiveDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, model_type: str, model_max_length: int):
        self.dataset = load_dataset('json', data_files=data_path, split='train')
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length

        if 'llama3' in model_type:
            print(f'llama3 dataset')
            self.system_format='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
            self.user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
            self.assistant_format='{content}<|eot_id|>\n'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Llama3 chat template: f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        data = self.dataset[idx]
        rag_input_ids, raw_input_ids, rag_labels, raw_labels = [], [], [], []

        rag_human  = self.user_format.format(content=data['rag_input'])
        rag_assistant = self.assistant_format.format(content=data['rag_output'])
        raw_human = self.user_format.format(content=data['raw_input'])
        raw_assistant = self.assistant_format.format(content=data['raw_output'])
        
        rag_input_tokens = self.tokenizer.encode(rag_human, add_special_tokens=False)
        rag_output_tokens = self.tokenizer.encode(rag_assistant, add_special_tokens=False)
        raw_input_tokens = self.tokenizer.encode(raw_human, add_special_tokens=False)
        raw_output_tokens = self.tokenizer.encode(raw_assistant, add_special_tokens=False)

        rag_input_ids = rag_input_tokens + rag_output_tokens
        rag_target_mask = [0] * len(rag_input_tokens) + [1] * len(rag_output_tokens)
        raw_input_ids = raw_input_tokens + raw_output_tokens
        raw_target_mask = [0] * len(raw_input_tokens) + [1] * len(raw_output_tokens)

        assert len(rag_input_ids) == len(rag_target_mask)
        assert len(raw_input_ids) == len(raw_target_mask)

        rag_input_ids = rag_input_ids[:self.model_max_length]
        raw_input_ids = raw_input_ids[:self.model_max_length]
        rag_target_mask = rag_target_mask[:self.model_max_length]
        raw_target_mask = raw_target_mask[:self.model_max_length]
        
        rag_attention_mask = [1] * len(rag_input_ids)
        raw_attention_mask = [1] * len(raw_input_ids)
        assert len(rag_input_ids) == len(rag_target_mask) == len(rag_attention_mask) 
        assert len(raw_input_ids) == len(raw_target_mask) == len(raw_attention_mask)
        rag_inputs = {
            'input_ids': rag_input_ids,
            'attention_mask': rag_attention_mask,
            'target_mask': rag_target_mask
        }
        raw_inputs = {
            'input_ids': raw_input_ids,
            'attention_mask': raw_attention_mask,
            'target_mask': raw_target_mask
        }
        return rag_inputs, raw_inputs

# 3. collator
class BothContravasiveCollator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, model_max_length: int):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features):
        rag_inputs = [f[0] for f in features]
        raw_inputs = [f[1] for f in features]

        rag_max_lengths = [len(x['input_ids']) for x in rag_inputs if x['input_ids'] is not None]
        rag_batch_max_len = min(max(rag_max_lengths), self.model_max_length)

        rag_input_ids_batch, rag_attention_mask_batch, rag_target_mask_batch = [], [], []        
        for x in rag_inputs:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = rag_batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            target_mask = target_mask[:self.model_max_length]

            rag_input_ids_batch.append(input_ids)
            rag_attention_mask_batch.append(attention_mask)
            rag_target_mask_batch.append(target_mask)

        rag_input_ids_batch = torch.tensor(rag_input_ids_batch, dtype=torch.long)
        rag_attention_mask_batch = torch.tensor(rag_attention_mask_batch, dtype=torch.long)
        rag_target_mask_batch = torch.tensor(rag_target_mask_batch, dtype=torch.long)
        
        rag_labels = torch.where(rag_target_mask_batch == 1, rag_input_ids_batch, -100)
        
        # Handle raw (parametric-memory) branch
        raw_max_lengths = [len(x['input_ids']) for x in raw_inputs if x['input_ids'] is not None]
        raw_batch_max_len = min(max(raw_max_lengths), self.model_max_length)

        # truncate and pad
        raw_input_ids_batch, raw_attention_mask_batch, raw_target_mask_batch = [], [], []        
        for x in raw_inputs:
            input_ids = x['input_ids']
            attention_mask = x['attention_mask']
            target_mask = x['target_mask']
            if input_ids is None:
                print('some input_ids is None')
                continue
            padding_len = raw_batch_max_len - len(input_ids)
            # padding
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # truncate
            input_ids = input_ids[:self.model_max_length]
            attention_mask = attention_mask[:self.model_max_length]
            target_mask = target_mask[:self.model_max_length]

            raw_input_ids_batch.append(input_ids)
            raw_attention_mask_batch.append(attention_mask)
            raw_target_mask_batch.append(target_mask)

        # Convert lists to final model-input tensors
        raw_input_ids_batch = torch.tensor(raw_input_ids_batch, dtype=torch.long)
        raw_attention_mask_batch = torch.tensor(raw_attention_mask_batch, dtype=torch.long)
        raw_target_mask_batch = torch.tensor(raw_target_mask_batch, dtype=torch.long)

        raw_labels = torch.where(raw_target_mask_batch == 1, raw_input_ids_batch, -100)
        return {
            'rag_input_ids': rag_input_ids_batch,
            'rag_attention_mask': rag_attention_mask_batch,
            'rag_labels': rag_labels,
            'raw_input_ids': raw_input_ids_batch,
            'raw_attention_mask':raw_attention_mask_batch,            
            'raw_labels': raw_labels
        }
