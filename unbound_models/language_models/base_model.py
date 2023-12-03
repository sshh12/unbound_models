from typing import List, Dict
from abc import ABC, abstractmethod

from torch.nn.functional import conv1d
import torch
import logging

from unbound_models.bindings.base_binding import BaseBinding


class UnboundMetaModel:
    def __init__(self, config):
        super(UnboundMetaModel, self).__init__(config)

    # def _load_projector_weights(self, weights: Dict):
    #     weights = {
    #         (k[23:] if k.startswith("base_model.model.model.") else k): v
    #         for k, v in weights.items()
    #     }
    #     logging.info(f"Loading pretrained weights: {list(weights.keys())}")
    #     load_result = self.load_state_dict(weights, strict=False)
    #     assert (
    #         len(load_result.unexpected_keys) == 0
    #     ), "Unexpected weights, is this the right model?"

    def initialize_pretrained_modules(
        self, modalities: List[BaseBinding], weights: Dict
    ):
        # for m in modalities:
        #     projector = m.build_projector(self.config.hidden_size)
        #     setattr(self, m.name + "_lmm_projector", projector)

        # self._load_projector_weights(weights)
        pass

    def initialize_modules(self, modalities: List[BaseBinding], weights: Dict):
        # names = [m.name for m in modalities]

        # self.config.modalities = names

        # for m in modalities:
        #     projector = m.build_projector(self.config.hidden_size)
        #     setattr(self, m.name + "_lmm_projector", projector)

        # self._load_projector_weights(weights)
        pass


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
