from unbound_models.language_models.mistral import (
    MistralUnboundForCausalLM,
)

LANGUAGE_MODEL_CLASSES = [MistralUnboundForCausalLM]

LANGUAGE_MODEL_NAME_TO_CLASS = {cls.__name__: cls for cls in LANGUAGE_MODEL_CLASSES}
