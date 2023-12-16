from typing import List, Optional, Tuple, Union

import torch

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    MistralConfig,
    MistralModel,
    MistralForCausalLM,
)

from unbound_models.language_models.base_model import (
    UnboundMetaModel,
    UnboundMetaForCausalLM,
    UnboundOutput,
)


class MistralUnboundConfig(MistralConfig):
    model_type = "mistral-unbound"


class MistralUnboundModel(UnboundMetaModel, MistralModel):
    config_class = MistralUnboundConfig

    def __init__(self, config: MistralUnboundConfig):
        super(MistralUnboundModel, self).__init__(config)


class MistralUnboundForCausalLM(MistralForCausalLM, UnboundMetaForCausalLM):
    config_class = MistralUnboundConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = MistralUnboundModel(config)

        self.vocab_size = config.vocab_size
        self.binding = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self) -> "MistralUnboundForCausalLM":
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, UnboundOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
            should_compute_loss,
        ) = self.prepare_inputs_labels_for_binding(
            input_ids, attention_mask, past_key_values, labels, **kwargs
        )
        assert input_ids is None
        assert labels is None
        assert inputs_embeds is not None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        loss = None
        if should_compute_loss:
            loss = self.binding.compute_loss(
                hidden_states,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs
            )

        if not return_dict:
            output = (None,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        resp = UnboundOutput(
            loss=loss,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            unbound_outputs=self.binding.compute_unbound_outputs(
                hidden_states, **kwargs
            ),
        )

        return resp

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        binding_inputs=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None:
            raise ValueError("inputs_embeds not supported")

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": None,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            **(binding_inputs or {}),
        }

        return model_inputs


AutoConfig.register("mistral-unbound", MistralUnboundConfig)
AutoModelForCausalLM.register(MistralUnboundConfig, MistralUnboundForCausalLM)
