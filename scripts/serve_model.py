from dataclasses import dataclass, field
import logging

from flask import Flask, request, jsonify
import transformers
import torch

from multi_token.training import (
    ModelArguments,
)
from multi_token.inference import load_trained_lora_model


@dataclass
class ServeArguments(ModelArguments):
    port: int = field(default=8080)
    host: str = field(default="0.0.0.0")
    load_bits: int = field(default=16)
    max_new_tokens: int = field(default=128)
    temperature: float = field(default=0.01)


# encoded = model.binding.encode_row(
#     {
#         "messages": [
#             {"role": "user", "content": "What is 1+1?"},
#             {"role": "assistant", "content": "1"},
#         ]
#     }
# )
# model_result = model.forward(
#     input_ids=None,
#     attention_mask=encoded["input_output_mask"].ne(0).unsqueeze(0).to(model.device),
#     **({"tokens_to_embed": encoded["tokens_to_embed"].unsqueeze(0).to(model.device)})
# )
# model.binding.compute_loss(
#     model_result.hidden_states, inputs_embeds=model_result.inputs_embeds
# )

# encoded = model.binding.encode_row({"messages": [{"role": "user", "content": "Hello"}]})
# prompt_emb = model.model.embed_tokens(encoded["tokens_to_embed"].to(model.device))[-5]
# bank = model.model.embed_tokens(torch.arange(32000).to(model.device))
# import torch

# model.binding.compute_loss(
#     model_result.hidden_states,
#     inputs_embeds=inputs_embeds,
#     attention_mask=attention_mask,
#     **kwargs
# )

# with torch.inference_mode():
#     model_result = model.forward(
#         input_ids=None,
#         attention_mask=encoded["input_output_mask"].ne(0).unsqueeze(0).to(model.device),
#         **(
#             {
#                 "tokens_to_embed": encoded["tokens_to_embed"]
#                 .unsqueeze(0)
#                 .to(model.device)
#             }
#         )
#     )

#     result_emb = model_result.bind_values[0, -1]
#     sim_emb = torch.linalg.norm(result_emb - bank, dim=1)
#     sim_known = torch.linalg.norm(prompt_emb - bank, dim=1)
#     print("emb", tokenizer.decode(sim_emb.argmin()))
#     print(sim_emb[sim_emb.argsort()])
#     print(sim_known[sim_known.argsort()])


# encoded = model.binding.encode_row(
#     {"messages": [{"role": "user", "content": "What is 1+1?"}]}
# )
# bank = model.model.embed_tokens(torch.arange(32000).to(model.device))

# with torch.inference_mode():
#     model_result = model.forward(
#         input_ids=None,
#         attention_mask=encoded["input_output_mask"].ne(0).unsqueeze(0).to(model.device),
#         **(
#             {
#                 "tokens_to_embed": encoded["tokens_to_embed"]
#                 .unsqueeze(0)
#                 .to(model.device)
#             }
#         )
#     )

#     result_emb = model_result.bind_values[0, -1]
#     loss = torch.nn.MSELoss(reduction="none")(result_emb.tile((32000, 1)), bank).mean(
#         axis=1
#     )
#     loss = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(result_emb.tile((32000, 1)), bank)

#     loss = torch.nn.PairwiseDistance(p=2.0, eps=1e-06)(
#         result_emb.tile((32000, 1)), bank
#     )
# sim_emb = torch.linalg.norm(result_emb - bank, dim=1)
# sim_known = torch.linalg.norm(prompt_emb - bank, dim=1)
# print("emb", tokenizer.decode(sim_emb.argmin()))
# print(sim_emb[sim_emb.argsort()])
# print(sim_known[sim_known.argsort()])

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

    import IPython

    IPython.embed()

    # @app.route("/generate", methods=["POST"])
    # def generate():
    #     req_json = request.get_json()

    #     encoded_dict = encode_chat(req_json, tokenizer, model.modalities)

    #     with torch.inference_mode():
    #         output_ids = model.generate(
    #             input_ids=encoded_dict["input_ids"].unsqueeze(0).to(model.device),
    #             max_new_tokens=serve_args.max_new_tokens,
    #             use_cache=True,
    #             do_sample=True,
    #             temperature=serve_args.temperature,
    #             modality_inputs={
    #                 m.name: [encoded_dict[m.name]] for m in model.modalities
    #             },
    #         )

    #     outputs = tokenizer.decode(
    #         output_ids[0, encoded_dict["input_ids"].shape[0] :],
    #         skip_special_tokens=True,
    #     ).strip()

    #     return jsonify({"output": outputs})

    # app.run(host=serve_args.host, port=serve_args.port)
