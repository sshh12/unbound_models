import argparse
import random
import numpy as np

import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from transformers import CLIPVisionModel, CLIPImageProcessor
from datasets import Dataset, load_dataset


def main(args):
    data = load_dataset("Falah/image_generation_prompts_SDXL", split="train")

    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")

    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    idxes = list(range(len(data)))
    idxes = random.sample(idxes, 20)
    batch_size = 5

    def gen():
        # iter idxes in batch_size
        for i in range(0, len(idxes), batch_size):
            batch = [data[idx]["prompts"][1:-1] for idx in idxes[i : i + batch_size]]

            with torch.no_grad():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(batch)

                out = pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    generator=torch.manual_seed(1),
                )

                image_prepro = image_processor(out.images)
                image_prepro["pixel_values"] = torch.Tensor(
                    np.array(image_prepro["pixel_values"])
                )
                clip_feats = image_model(**image_prepro).pooler_output

            for i, prompt in enumerate(batch):
                yield {
                    "prompt": prompt,
                    "prompt_embeds": prompt_embeds[i].cpu().numpy(),
                    "negative_prompt_embeds": negative_prompt_embeds[i].cpu().numpy(),
                    "pooled_prompt_embeds": pooled_prompt_embeds[i].cpu().numpy(),
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds[i]
                    .cpu()
                    .numpy(),
                    "clip": clip_feats[i].cpu().numpy(),
                }

    ds = Dataset.from_generator(gen)
    ds.save_to_disk(args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--output_folder", type=str, default="/data/reverse-diffusion"
    )
    args = parser.parse_args()
    main(args)
