from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from text_utils import format_prompt, extract_json

# Function to Load LLM on a Specific GPU
def load_llm_on_gpu(gpu_id, model_name, batch_size):
    print(f"Loading LLM on GPU {gpu_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Enable 8-bit or 4-bit quantization
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model on the assigned GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": gpu_id}, 
        trust_remote_code=True
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size 
    )

def process_on_gpu(gpu_id, texts, batch_size):
    nlp_model = load_llm_on_gpu(gpu_id)

    results = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        # Generate prompts
        prompts = [format_prompt(text) for text in batch_texts]

        responses = nlp_model(prompts, max_length=1000, do_sample=False)

        for prompt, response in zip(prompts, responses):
            try:
                full_response = response[0]["generated_text"]
            except (IndexError, KeyError):
                full_response = "ERROR: LLM response missing"

            # Extract direct objects and verbs
            direct_objects, verbs = extract_json(full_response)

            results.append((prompt, full_response, direct_objects, verbs))

    return results