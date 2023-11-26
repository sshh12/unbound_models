import transformers
import logging

from unbound_models.training import (
    TrainingArguments,
    ModelArguments,
    train_for_binding,
)
from unbound_models.training_data import (
    DataArguments,
)
from unbound_models.language_models import LANGUAGE_MODEL_NAME_TO_CLASS
from unbound_models.bindings import BINDING_BUILDERS

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser(
        (TrainingArguments, ModelArguments, DataArguments)
    )

    training_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    binding = BINDING_BUILDERS[model_args.binding_builder]()
    model_cls = LANGUAGE_MODEL_NAME_TO_CLASS[model_args.model_cls]

    train_for_binding(model_cls, training_args, model_args, data_args, binding)
