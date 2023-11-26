from typing import Dict

from transformers import AutoTokenizer
import torch

from unbound_models.bindings.base_binding import BaseBinding

TOKENIZER = "mistralai/Mistral-7B-Instruct-v0.1"
SEP = "[/INST]"


class TextTokenBinding(BaseBinding):
    def __init__(
        self,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        self.embed_tokens = None

    def encode_row(self, row: Dict) -> Dict:
        convo_chat = self.tokenizer.apply_chat_template(row["messages"], tokenize=False)
        tokens_input = self.tokenizer(
            convo_chat.split(SEP)[0] + SEP, add_special_tokens=False
        )
        tokens_output = self.tokenizer(
            convo_chat.split(SEP)[1], add_special_tokens=False
        )
        embs = self.embed_tokens(
            torch.cat(tokens_input["input_ids"]), tokens_output["input_ids"]
        )
        mask = torch.Tensor(
            [0] * len(tokens_input["input_ids"]) + [1] * len(tokens_output["input_ids"])
        )
        return {"token_embeddings": embs, "output_mask": mask}
