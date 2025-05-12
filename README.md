# Llama3 C# Fine-Tuning Project

This repository contains the Jupyter Notebook for fine-tuning the Llama 3.1 (8B) model on C# code generation tasks using the [Azamorn/tiny-codes-csharp](https://huggingface.co/datasets/Azamorn/tiny-codes-csharp) dataset and Unsloth's PEFT (LoRA) pipeline.

### Usage Example

```
from unsloth import FastLanguageModel
from transformers import TextStreamer

model, tokenizer = FastLanguageModel.from_pretrained(
  model_name="umutakman/llama3.1-8b-finetuned-csharp",
  load_in_4bit=True
)
model = FastLanguageModel.for_inference(model)
text = "Build an RPG console application"
prompt = alpaca_prompt.format(
        "Build an RPG console application.", # instruction
        "" # output (leave empty)
    )
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
streamer = TextStreamer(tokenizer)
model.generate(**inputs, streamer=streamer, max_new_tokens=512)
```

[llama3.1-8b-finetuned-csharp](https://huggingface.co/umutakman/llama3.1-8b-finetuned-csharp)

[llama3.1-8b-finetuned-csharp-q4](https://huggingface.co/umutakman/llama3.1-8b-finetuned-csharp-q4)



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
  https://colab.research.google.com/github/umutakkman/llama3-csharp-finetune/blob/main/Generative_AI_Fine_Tuning_using_Sloth_Llama3_1_(8B).ipynb
)
