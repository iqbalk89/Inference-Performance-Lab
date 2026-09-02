"""A small, readable version of the Week 1 PyTorch inference loop.

Run this on the Lambda GPU after the environment setup in README.md:

    .venv/bin/python experiments/agent-prefix-performance/week1_pytorch_minimal.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Hugging Face downloads this model the first time and reuses the local cache
# on later runs. This is Transformers code, not vLLM.
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


# A short prompt keeps this teaching example easy and inexpensive to run.
PROMPT = "Explain why a KV cache makes autoregressive generation faster."


def main() -> None:
    # The model must be loaded onto the GPU before CUDA can execute its layers.
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("This example requires CUDA-enabled PyTorch and an NVIDIA GPU.")

    # The tokenizer converts text into integer token IDs.
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    # Transformers constructs the Qwen model. PyTorch stores its parameters as
    # tensors, and .to(device) copies those tensors into A10 VRAM.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()  # Use inference behavior, for example disable dropout.

    # Move the token IDs and attention mask from CPU memory to GPU memory too.
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    print(f"input_ids shape: {tuple(inputs['input_ids'].shape)}")

    # ------------------------------
    # Prefill: process the whole prompt in one forward pass.
    # ------------------------------
    with torch.inference_mode():
        prefill = model(**inputs, use_cache=True)

    # logits contains a score for every vocabulary token at every prompt
    # position. We use the final prompt position to choose the first output.
    print(f"logits shape: {tuple(prefill.logits.shape)}")
    next_token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # past_key_values contains the K and V tensors created by every layer.
    # Keeping them avoids recomputing the prompt during the next decode step.
    kv_cache = prefill.past_key_values
    print(f"cache type: {type(kv_cache).__name__}")

    # ------------------------------
    # Decode: generate one token at a time using the existing cache.
    # ------------------------------
    generated_tokens = [next_token]
    attention_mask = inputs["attention_mask"]

    with torch.inference_mode():
        for _ in range(31):
            # The model receives only the newest token, not the whole prompt.
            attention_mask = torch.cat(
                (attention_mask, attention_mask.new_ones((1, 1))),
                dim=1,
            )
            step = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=kv_cache,
                use_cache=True,
            )

            # The new output cache contains the old history plus this position.
            kv_cache = step.past_key_values
            next_token = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)

    # Convert generated token IDs back into readable text.
    output_ids = torch.cat(generated_tokens, dim=1)
    print("generated text:")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
