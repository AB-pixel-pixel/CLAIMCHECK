from vllm import LLM, SamplingParams


def main():
    model = "Qwen/Qwen3.5-4B"
    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=8192,
        disable_log_stats=True,
        enforce_eager=True,
    )
    outputs = llm.generate(
        ["Say only OK."],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    print(outputs[0].outputs[0].text)


if __name__ == "__main__":
    main()
