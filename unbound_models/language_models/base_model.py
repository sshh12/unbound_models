from typing import Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from transformers.modeling_outputs import CausalLMOutputWithPast

from unbound_models.bindings.base_binding import BaseBinding


@dataclass
class UnboundOutput(CausalLMOutputWithPast):
    unbound_outputs: Dict = None


class UnboundMetaModel:
    def __init__(self, config):
        super(UnboundMetaModel, self).__init__(config)


class UnboundMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self) -> "UnboundMetaForCausalLM":
        pass

    def prepare_inputs_labels_for_binding(
        self, input_ids, attention_mask, past_key_values, labels, **kwargs
    ):
        model = self.get_model()
        return self.binding.prepare_inputs(
            model, attention_mask, past_key_values, **kwargs
        )
