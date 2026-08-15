# finetuning-llama
# Llama 2 7B Fine-Tuning with QLoRA

##  Project Overview

This project demonstrates the fine-tuning of **Llama 2 7B Chat** using **QLoRA (Quantized Low-Rank Adaptation)** on the **Guanaco instruction dataset**.

The goal is to adapt a pretrained Large Language Model (LLM) to better follow user instructions while reducing GPU memory requirements through **4-bit quantization and LoRA adapters**.

##  Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Hugging Face TRL
* PEFT
* BitsAndBytes
* Llama 2 7B
* QLoRA / LoRA

## Model & Dataset

**Base Model:** `NousResearch/Llama-2-7b-chat-hf`

**Dataset:** `mlabonne/guanaco-llama2-1k`

The Guanaco dataset contains instruction-response examples that are used to fine-tune the model for conversational and instruction-following tasks.

##  Fine-Tuning Configuration

| Parameter             | Value              |
| --------------------- | ------------------ |
| Quantization          | 4-bit NF4          |
| LoRA Rank             | 64                 |
| LoRA Alpha            | 16                 |
| LoRA Dropout          | 0.1                |
| Learning Rate         | 2e-4               |
| Epochs                | 1                  |
| Batch Size            | 1                  |
| Gradient Accumulation | 4                  |
| Precision             | FP16               |
| Optimizer             | Paged AdamW 32-bit |

##  Training Process

The project uses **QLoRA** to reduce memory usage during fine-tuning.

```text
Llama 2 7B
    ↓
4-bit NF4 Quantization
    ↓
LoRA Adapters
    ↓
Guanaco Instruction Dataset
    ↓
Fine-Tuning
    ↓
Fine-Tuned Model
    ↓
Text Generation
```

Only the LoRA adapter parameters are trained instead of updating all parameters of the 7B model.

##  Training Results

The model was trained for **1 epoch and 250 steps**.

**Final Training Loss:** approximately `1.276`

The training was performed on a GPU with approximately **14.5 GB VRAM**, demonstrating memory-efficient LLM fine-tuning using QLoRA.

##  Example

**Prompt:**

```text
Who is Cristiano Ronaldo?
```

The fine-tuned model generates an instruction-based response using the learned patterns from the training dataset.

##  Project Structure

```text
Llama2-QLoRA-FineTuning/
│
├── Llama2_FineTuning.ipynb
├── README.md
└── requirements.txt
```

##  How to Run

Install the required libraries:

```bash
pip install torch transformers datasets accelerate peft bitsandbytes trl
```

Open the `.ipynb` notebook in **Google Colab** or another Jupyter environment with a CUDA-enabled GPU and execute the cells sequentially.

##  Learning Outcomes

This project provides practical experience with:

* Large Language Model fine-tuning
* Llama 2
* QLoRA and LoRA
* 4-bit quantization
* Hugging Face Transformers and TRL
* PEFT
* GPU memory optimization
* LLM inference and text generation

##  Future Improvements

* Train with a larger dataset
* Perform multiple training epochs
* Evaluate the model using benchmark datasets
* Build a chatbot interface using Streamlit
* Deploy the fine-tuned model as an API
* Compare the base and fine-tuned models

##  Author

**Kartik Chakane**

