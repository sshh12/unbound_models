from typing import Dict, Sequence, Tuple

from transformers import AutoTokenizer
import torch.nn as nn
import torch

from unbound_models.bindings.base_binding import BaseBinding
from unbound_models.model_utils import fix_tokenizer, get_peft_state_non_lora

TOKENIZER = "mistralai/Mistral-7B-Instruct-v0.1"
SEP = "[/INST]"
HIDDEN_SIZE = 4096


class TextTokenBinding(BaseBinding):
    def __init__(
        self,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        fix_tokenizer(self.tokenizer)

        self.token_embedding_head = nn.Sequential(
            *[nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)]
        )
        self.device = "cuda"
        self.dtype = torch.float32

    def to(self, dtype: torch.dtype, device: torch.device):
        self.device = device
        self.dtype = dtype
        self.token_embedding_head.to(dtype=dtype, device=device)

    def encode_row(self, row: Dict) -> Dict:
        convo_chat = self.tokenizer.apply_chat_template(row["messages"], tokenize=False)
        tokens_input = self.tokenizer(
            convo_chat.split(SEP)[0] + SEP, add_special_tokens=False
        )
        tokens_output = self.tokenizer(
            convo_chat.split(SEP)[1], add_special_tokens=False
        )
        tokens_to_embed = torch.IntTensor(
            tokens_input["input_ids"] + tokens_output["input_ids"],
        )
        output_mask = torch.IntTensor(
            [0] * len(tokens_input["input_ids"]) + [1] * len(tokens_output["input_ids"])
        )
        input_output_mask = torch.IntTensor(
            [1] * len(tokens_input["input_ids"]) + [1] * len(tokens_output["input_ids"])
        )
        return {
            "tokens_to_embed": tokens_to_embed,
            "output_mask": output_mask,
            "input_output_mask": input_output_mask,
        }

    def collate_rows(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        tokens_to_embed, output_mask, input_output_mask = tuple(
            [instance[key] for instance in instances]
            for key in ["tokens_to_embed", "output_mask", "input_output_mask"]
        )

        tokens_to_embed = torch.nn.utils.rnn.pad_sequence(
            tokens_to_embed,
            batch_first=True,
            padding_value=0,
        )
        tokens_to_embed = tokens_to_embed[:, : self.tokenizer.model_max_length]

        output_mask = torch.nn.utils.rnn.pad_sequence(
            output_mask, batch_first=True, padding_value=0
        )
        output_mask = output_mask[:, : self.tokenizer.model_max_length]

        input_output_mask = torch.nn.utils.rnn.pad_sequence(
            input_output_mask, batch_first=True, padding_value=0
        )
        input_output_mask = input_output_mask[:, : self.tokenizer.model_max_length]

        batch = dict(
            tokens_to_embed=tokens_to_embed,
            output_mask=output_mask,
            attention_mask=input_output_mask.ne(0),
            input_ids=None,
            labels=None,
        )
        return batch

    def prepare_inputs(self, model, attention_mask, past_key_values, **kwargs) -> Tuple:
        inputs_embeds = model.embed_tokens(kwargs["tokens_to_embed"])

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
        token_embeddings = self.token_embedding_head(hidden_states)

        mask = kwargs["output_mask"][
            ...,
            1:,
        ].unsqueeze(-1)

        # Shift so that tokens < n predict n
        shift_pred_emb = token_embeddings[..., :-1, :].contiguous() * mask
        shift_true_emb = kwargs.get("inputs_embeds")[..., 1:, :].contiguous() * mask
        # Flatten the tokens

        shift_pred_emb = shift_pred_emb.view(-1, HIDDEN_SIZE)
        shift_true_emb = shift_true_emb.view(-1, HIDDEN_SIZE)
        # Enable model parallelism
        shift_true_emb = shift_true_emb.to(shift_pred_emb.device)

        loss_fct = nn.MSELoss()
        loss = loss_fct(shift_pred_emb, shift_true_emb)

        return loss

    def to_state_dict(self):
        return get_peft_state_non_lora(self.token_embedding_head.named_parameters())
