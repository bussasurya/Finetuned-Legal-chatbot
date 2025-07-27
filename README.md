
---

# ⚖️🇮🇳 Fine-Tuned Legal Chatbot

## **Fine-Tuning LLaMA 8B for Indian Legal Q\&A**

A powerful, low-resource legal assistant trained on Indian law (IPC, CrPC, Constitution, etc.) using **DeepSeek LLaMA 8B** with **Unsloth LoRA**.
This project leverages **Retrieval-Augmented Generation (RAG)** with **FAISS** and **Sentence-BERT** for accurate legal query responses.

---

## 🚀 Features

* 🧠 Fine-tuned **DeepSeek LLaMA-8B** model on Indian legal data using **Unsloth** and **LoRA**
* 🔍 Integrated **RAG pipeline** using **FAISS** and **Sentence-BERT**
* ⚡ Fast and memory-efficient training via **8-bit AdamW optimizer**, **PEFT**, and **Unsloth**
* ✅ **40% reduction in hallucinations** and improved legal factual consistency
* 🧪 Ready-to-use **inference script** for accurate and fast responses

---

## 📂 Dataset

Custom curated Indian legal datasets used for fine-tuning:

* `constitution_qa.json` – Questions from **Indian Constitution**
* `crpc_qa.json` – **Criminal Procedure Code** related Q\&A
* `ipc_qa.json` – **Indian Penal Code** Q\&A pairs
* `synthetic_legal_qa_dataset_unique_questions.jsonl` – High-quality **synthetic Q\&A** pairs across multiple Indian legal domains

---

## 🚀 Technologies Used

**Model & Training**
`PyTorch`, `Hugging Face Transformers`, `LoRA (PEFT)`, `Unsloth`

**RAG & Retrieval**
`FAISS`, `Sentence-BERT`

**Optimization**
`SFTTrainer`, `AdamW 8-bit`, `Gradient Checkpointing`

**Experiment Tracking**
`Weights & Biases (W&B)`

---

## 📈 Results & Improvements

| Metric                    | Before Fine-Tuning | After Fine-Tuning    |
| ------------------------- | ------------------ | -------------------- |
| Factual Accuracy (Eval)   | \~58%              | \~81%                |
| Hallucination Rate        | High               | Reduced by 40%       |
| Training Speed (per step) | Standard           | 2× faster (Unsloth)  |
| Memory Footprint (8B)     | \~35 GB            | <15 GB (LoRA + 8bit) |

---

## 📌 Inference Example

```python
prompt = (
    "### Instruction:\\n"
    "What rights does a person have if arrested without a warrant in India?\\n\\n"
    "### Response:"
)

inputs = tokenizer(prompt, return_tensors="pt").to(torch.cuda.current_device())

with torch.no_grad():
    output_ids = model.generate(
        inputs["input_ids"],
        max_length=512,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

---


