import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

from direct_obj_extractor.text_utils import format_prompt, extract_json


def load_llm_on_gpu(gpu_id, model_name):
    """ Load the LLM on the assigned GPU with efficient settings. """
    print(f"Loading LLM on GPU {gpu_id}...")

    torch.cuda.set_device(gpu_id)  # Ensure correct GPU is used

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Enable 8-bit quantization to reduce memory usage
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model on the assigned GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

def process_on_gpu(model_name, gpu_id, texts, batch_size):
    """ Process text in parallel on the assigned GPU """
    torch.cuda.set_device(gpu_id)  # Ensure correct GPU usage

    nlp_model = load_llm_on_gpu(gpu_id, model_name)  # Load once per process

    # Generate prompts in bulk (vectorized processing)
    prompts = [format_prompt(text) for text in texts]

    # Process all prompts at once in parallel
    responses = nlp_model(prompts, max_length=1000, do_sample=False, batch_size=batch_size)

    results = []
    for prompt, response in zip(prompts, responses):
        try:
            full_response = response[0]["generated_text"]
        except (IndexError, KeyError):
            full_response = "ERROR: LLM response missing"

        # Extract direct objects and verbs
        direct_objects, verbs = extract_json(full_response)

        results.append((prompt, full_response, direct_objects, verbs))

    return results

