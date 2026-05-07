import openai
import os
# import ollama # Removed as per user request
import torch
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import base64
from time import perf_counter

# Global variables for model and tokenizer to avoid reloading
_model = None
_tokenizer = None
_model_device = None

PAPER_ALIGN = os.getenv("CLAIMCHECK_PAPER_ALIGN", "1") != "0"
PAPER_MODEL = os.getenv("PAPER_LLM_MODEL", "Qwen/Qwen3-4B")
DEFAULT_LOCAL_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen3.5-4B")
DEFAULT_LOCAL_ADAPTER = os.getenv("LOCAL_LLM_ADAPTER", None)
DEFAULT_CUDA_DEVICE = os.getenv("LOCAL_CUDA_DEVICE", "1")


def _compact_prompt_text(prompt: str) -> str:
    max_chars = int(os.getenv("LOCAL_MAX_PROMPT_CHARS", "16000"))
    head_chars = int(os.getenv("LOCAL_PROMPT_HEAD_CHARS", "4000"))
    if not isinstance(prompt, str) or len(prompt) <= max_chars:
        return prompt
    tail_chars = max(max_chars - head_chars - 32, 0)
    if tail_chars <= 0:
        return prompt[:max_chars]
    return (
        prompt[:head_chars].rstrip()
        + "\n\n... [prompt truncated] ...\n\n"
        + prompt[-tail_chars:].lstrip()
    )


def _resolve_local_snapshot_path(model_name: str) -> str:
    cache_root = os.path.expanduser("~/.cache/huggingface/hub")
    repo_dir = os.path.join(cache_root, f"models--{model_name.replace('/', '--')}")
    snapshots_dir = os.path.join(repo_dir, "snapshots")
    if os.path.isdir(snapshots_dir):
        snapshots = sorted(
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        )
        if snapshots:
            return os.path.join(snapshots_dir, snapshots[-1])
    return model_name


def resolve_model_device():
    if not torch.cuda.is_available():
        return "cpu"

    try:
        device_index = int(DEFAULT_CUDA_DEVICE)
    except ValueError:
        device_index = 0

    if device_index < 0 or device_index >= torch.cuda.device_count():
        device_index = 0

    return f"cuda:{device_index}"


def get_model_and_tokenizer(model_name=None, adapter_path=None):
    global _model, _tokenizer, _model_device
    if PAPER_ALIGN:
        model_name = PAPER_MODEL
    else:
        model_name = model_name or DEFAULT_LOCAL_MODEL
    adapter_path = adapter_path or DEFAULT_LOCAL_ADAPTER
    if _model is None:
        _model_device = resolve_model_device()
        print(f"Loading local model: {model_name}...")
        print(f"Using model device: {_model_device}", flush=True)

        def _load_model(load_name, load_adapter_path, local_files_only=False):
            local_name = _resolve_local_snapshot_path(load_name) if local_files_only else load_name
            tokenizer = AutoTokenizer.from_pretrained(
                local_name,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.bfloat16,
                "local_files_only": local_files_only,
            }
            if _model_device.startswith("cuda"):
                model_kwargs["device_map"] = {"": _model_device}
            model = AutoModelForCausalLM.from_pretrained(local_name, **model_kwargs)
            if load_adapter_path and os.path.isdir(load_adapter_path):
                print(f"Loading LoRA adapter from: {load_adapter_path}")
                model = PeftModel.from_pretrained(model, load_adapter_path)
                print("LoRA adapter loaded successfully.")
            if _model_device == "cpu":
                model = model.to(_model_device)
            return tokenizer, model

        try:
            effective_adapter = None if PAPER_ALIGN else adapter_path
            _tokenizer, _model = _load_model(model_name, effective_adapter, local_files_only=True)
            print("Model loaded successfully.")
        except Exception as e:
            if PAPER_ALIGN and model_name == PAPER_MODEL:
                fallback_name = DEFAULT_LOCAL_MODEL
                fallback_adapter = adapter_path if os.path.isdir(adapter_path or "") else None
                print(f"Paper model unavailable locally; falling back to {fallback_name}.")
                try:
                    _tokenizer, _model = _load_model(fallback_name, fallback_adapter, local_files_only=True)
                    print("Fallback model loaded successfully.")
                except Exception as fallback_error:
                    print(f"Error loading fallback model {fallback_name}: {fallback_error}")
                    raise fallback_error
            else:
                print(f"Error loading model {model_name}: {e}")
                raise e
    return _model, _tokenizer, _model_device

def prompt_gpt(prompt, model='o4-mini-2025-04-16'):
    openai.api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    client = openai.OpenAI(api_key=openai.api_key, base_url="https://xiaoai.plus/v1")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def prompt_local(
    prompt,
    model=None,
    think=True,
    use_adapter=False,
    max_new_tokens=None,
    temperature=None,
    do_sample=None,
    top_p=None,
):
    """
    Generates a response using the local Hugging Face model.
    Replaces the previous prompt_ollama function.
    """
    # specific handling for "think" parameter if needed, 
    # though standard Qwen doesn't use it in the same way as reasoning models might.
    # We can add a system prompt if 'think' is False to discourage reasoning if the model supports it,
    # but for now we'll stick to the standard chat template.
    force_think = os.getenv("CLAIMCHECK_FORCE_THINK", "").strip().lower()
    if force_think in {"1", "true", "yes", "on"}:
        think = True
    elif force_think in {"0", "false", "no", "off"}:
        think = False

    model = model or DEFAULT_LOCAL_MODEL
    model_instance, tokenizer, model_device = get_model_and_tokenizer(model)
    prompt = _compact_prompt_text(prompt)
    prompt_chars = len(prompt) if isinstance(prompt, str) else 0
    print(f"[LLM] prompt_local start | model={model} | think={think} | prompt_chars={prompt_chars}", flush=True)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    max_input_tokens = int(os.getenv("LOCAL_MAX_INPUT_TOKENS", "4096"))
    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    model_inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model_device)
    tokenizer.truncation_side = original_truncation_side
    
    # Generate response
    # You might want to adjust max_new_tokens or other parameters
    t0 = perf_counter()
    print("[LLM] generating response...", flush=True)
    if max_new_tokens is None:
        max_new_tokens = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "512"))
    if do_sample is None:
        do_sample = True
    if temperature is None:
        temperature = 0.6
    if top_p is None:
        top_p = 0.95

    generation_context = nullcontext()
    if isinstance(model_instance, PeftModel) and not use_adapter:
        generation_context = model_instance.disable_adapter()

    with torch.inference_mode():
        with generation_context:
            generated_ids = model_instance.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
            )
    t1 = perf_counter()
    print(f"[LLM] generation done | elapsed_sec={t1 - t0:.2f}", flush=True)
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Strip <think> blocks when the template returns reasoning traces.
    if "</think>" in response:
        response = response.split("</think>", 1)[-1].strip()

    del generated_ids
    del model_inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[LLM] prompt_local end | response_chars={len(response)}", flush=True)
    return response
