from typing import List, Dict, Sequence
from dataclasses import dataclass, field
import os

from torch.utils.data import Dataset
from datasets import load_from_disk, load_dataset, Dataset as HFDataset
import transformers
import torch

from unbound_models.bindings.base_binding import BaseBinding


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
        return self.binding.collate_rows(instances)
