from typing import List
import argparse

from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

from unbound_models.constants import ROLE_ASSISTANT, ROLE_USER

SEP = "[/INST]"


def _convert_convo(convo) -> List:
    msgs = [
        {
            "role": ROLE_USER,
            "content": convo["instruction"],
        },
        {
            "role": ROLE_ASSISTANT,
            "content": convo["output"],
        },
    ]
    return msgs


def main(args):
    dataset = load_dataset(args.dataset, data_files="*.json", split="train")

    def gen(rows):
        for idx in rows:
            convo = _convert_convo(dataset[idx])
            yield {
                "messages": convo,
            }

    ds = Dataset.from_generator(
        gen, gen_kwargs={"rows": list(range(len(dataset)))}, num_proc=args.num_proc
    )
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="teknium/openhermes")
    parser.add_argument(
        "-o", "--output_folder", type=str, default="/data/text-token-hermes-dataset"
    )
    parser.add_argument("-n", "--num_proc", type=int, default=1)
    args = parser.parse_args()
    main(args)
