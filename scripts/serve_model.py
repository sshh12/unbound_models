from dataclasses import dataclass, field
import logging

from flask import Flask, request, jsonify
import transformers
import torch

from unbound_models.training import (
    ModelArguments,
)
from unbound_models.inference import load_trained_lora_model


@dataclass
class ServeArguments(ModelArguments):
    port: int = field(default=8080)
    host: str = field(default="0.0.0.0")
    load_bits: int = field(default=16)
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.01)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = transformers.HfArgumentParser((ServeArguments,))

    serve_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    model, tokenizer = load_trained_lora_model(
        model_name_or_path=serve_args.model_name_or_path,
        model_lora_path=serve_args.model_lora_path,
        load_bits=serve_args.load_bits,
    )

    app = Flask(__name__)

    @app.route("/generate", methods=["POST"])
    def generate():
        req_json = request.get_json()

        last_gen = None
        with torch.inference_mode():
            for gen_out in model.binding.generate(
                model.forward,
                dict(req_json),
            ):
                print("generate =", gen_out)
                last_gen = gen_out

        return jsonify(last_gen)

    app.run(host=serve_args.host, port=serve_args.port)
