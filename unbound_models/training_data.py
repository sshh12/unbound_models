from typing import List, Dict, Sequence
from dataclasses import dataclass, field
import os

from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch

from unbound_models.bindings.base_binding import BaseBinding
from unbound_models.constants import IGNORE_INDEX


@dataclass
class DataArguments:
    dataset_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


def _resolve_dataset(path: str) -> HFDataset:
    if os.path.exists(path):
        return load_from_disk(path)
    else:
        return load_dataset(path, split="train", data_files="*.arrow")


class UnboundDataset(Dataset):
    def __init__(
        self,
        data_args: DataArguments,
        tokenizer: transformers.PreTrainedTokenizer,
        binding: List[BaseBinding],
    ):
        super(UnboundDataset, self).__init__()
        self.dataset = _resolve_dataset(data_args.dataset_path)
        self.tokenizer = tokenizer
        self.binding = binding

    def __len__(self):
        return len(self.dataset)

    def get_example(self) -> Dict:
        return self.dataset[0]

    def __getitem__(self, i) -> Dict:
        item = self.dataset[i]
        return self.binding.encode_row(item)


@dataclass
class DataCollatorForSupervisedUnboundDataset:
    tokenizer: transformers.PreTrainedTokenizer
    binding: List[BaseBinding]

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ["input_ids", "labels"]
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        for m in self.modalities:
            batch[m.name] = [instance[m.name] for instance in instances]

        return batch
