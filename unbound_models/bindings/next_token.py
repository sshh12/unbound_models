from typing import Dict, Sequence, Tuple, List, Callable, Generator

from transformers import AutoTokenizer
import torch.nn as nn
import torch

from unbound_models.bindings.base_binding import BaseBinding
from unbound_models.model_utils import fix_tokenizer, get_trained_state_non_lora

TOKENIZER = "mistralai/Mistral-7B-Instruct-v0.1"
SEP = "[/INST]"


class NextTokenBinding(BaseBinding):
    def __init__(
        self, model_hidden_size: int = 4096, logit_model_hidden_size: int = 4096
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        fix_tokenizer(self.tokenizer)

        self.logits_model = nn.Sequential(
            *[
                nn.Linear(model_hidden_size, logit_model_hidden_size),
                nn.GELU(),
                nn.Linear(
                    logit_model_hidden_size, self.tokenizer.vocab_size, bias=False
                ),
            ]
        )
        self.device = "cuda"
        self.dtype = torch.float32

    def to(self, dtype: torch.dtype, device: torch.device):
        self.device = device
        self.dtype = dtype
        self.logits_model.to(dtype=dtype, device=device)

    def encode_row(self, row: Dict) -> Dict:
        convo_chat = self.tokenizer.apply_chat_template(row["messages"], tokenize=False)
        tokens_input = self.tokenizer(
            convo_chat.split(SEP)[0] + SEP, add_special_tokens=False
        )
        tokens_output = self.tokenizer(
            convo_chat.split(SEP)[1], add_special_tokens=False
        )
        tokens = torch.LongTensor(
            tokens_input["input_ids"] + tokens_output["input_ids"],
        )
        output_mask = torch.IntTensor(
            [0] * len(tokens_input["input_ids"]) + [1] * len(tokens_output["input_ids"])
        )
        input_output_mask = torch.IntTensor(
            [1] * len(tokens_input["input_ids"]) + [1] * len(tokens_output["input_ids"])
        )
        return {
            "tokens": tokens,
            "output_mask": output_mask,
            "input_output_mask": input_output_mask,
        }

    def collate_rows(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert len(instances) == 1

        tokens, output_mask, input_output_mask = tuple(
            instances[0][key][: self.tokenizer.model_max_length].unsqueeze(0)
            for key in ["tokens", "output_mask", "input_output_mask"]
        )

        batch = dict(
            tokens=tokens,
            output_mask=output_mask,
            attention_mask=input_output_mask.ne(0),
            input_ids=None,
            labels=None,
        )
        return batch

    def prepare_inputs(self, model, attention_mask, past_key_values, **kwargs) -> Tuple:
        inputs_embeds = model.embed_tokens(kwargs["tokens"])

        should_compute_loss = "output_mask" in kwargs
        return (
            None,
            attention_mask,
            past_key_values,
            inputs_embeds,
            None,
            should_compute_loss,
        )

    def compute_loss(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.logits_model(hidden_states)
        labels = kwargs["tokens"].to(logits.device)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.tokenizer.vocab_size)
        shift_labels = shift_labels.view(-1)

        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return loss

    def compute_unbound_outputs(
        self, hidden_states: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        logits = self.logits_model(hidden_states)
        return {"logits": logits}

    def named_parameters(self) -> List:
        return list(self.logits_model.named_parameters())

    def to_state_dict(self):
        return get_trained_state_non_lora(self.named_parameters())

    def load_state_dict(self, weights: Dict):
        self.logits_model.load_state_dict(weights, strict=True)

    def generate(
        self, forward: Callable, params: Dict, device: str = "cuda"
    ) -> Generator[Dict, None, None]:
        max_length = params.pop("max_length", 128)
        encoded = self.encode_row(params)

        attention_mask = encoded["input_output_mask"].ne(0).unsqueeze(0).to(device)
        tokens = encoded["tokens"].unsqueeze(0).to(device)

        output_tokens = []

        while len(output_tokens) < max_length:
            model_output = forward(
                input_ids=None,
                attention_mask=attention_mask,
                **({"tokens": tokens}),
            )
            next_token_id = model_output.unbound_outputs["logits"][0, -1].argmax().cpu()

            attention_mask = torch.cat(
                [attention_mask, torch.BoolTensor([[True]]).to(device)], dim=1
            )

            tokens = torch.cat(
                [tokens, torch.IntTensor([[next_token_id]]).to(device)], dim=1
            )

            if next_token_id == self.tokenizer.eos_token_id:
                break
            output_tokens.append(next_token_id)
            char = self.tokenizer.decode(next_token_id)
            yield {
                "output": self.tokenizer.decode(output_tokens).strip(),
                "next_token": char,
            }
