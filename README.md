# LightFED_MVQA

A Federated Learning Framework using Small Language Models with Multimodal Retrieval-Augmented Generation for Medical Visual Question Answering

This project implements a Federated Learning (FL) system integrated with Retrieval-Augmented Generation (RAG) to tackle the Medical Visual Question Answering (Med-VQA) task.

The system allows medical institutions (Hospitals/Research Institutes) to collaboratively train a multimodal Small Language Model (SLM) — specifically Qwen2-VL (2B) — without sharing sensitive patient data externally. Concurrently, the diagnostic accuracy is significantly improved through the retrieval of similar medical cases from a local Vector Database (FAISS).


## Key Features

* **Privacy Preservation (Federated Learning):** Simulates distributed training across multiple institutions (Clients) using the FedAvg algorithm. Patient data strictly remains on local servers.
* **Shared-Engine Architecture (RAM/VRAM Optimization):** Initializes the core model engine exactly once in memory. Virtual clients only swap LoRA (Low-Rank Adaptation) weights, completely eliminating Out-Of-Memory (OOM) errors when running simulations on a personal computer.
* **Medical RAG (Retrieval-Augmented Generation):** Integrates FAISS to search for similar medical cases/X-ray images as context to "prompt" the model before it generates the final answer.
* **Comprehensive Baseline System:** Built-in scripts to run and compare performance across:
    - Centralized + RAG: Centralized training with RAG (Upper Bound).
    - Proposed (Fed + RAG): The proposed architecture (Federated Learning + RAG).
    - Fed-SLM (No RAG): Standard FL model without RAG.
    - Fed-LLaVA-Med (13B): Popular baseline used in Q1/Q2 tier papers.

## System Requirements

- **Operating System:** Linux (Ubuntu) / Windows (WSL2 recommended).
- **Hardware:** NVIDIA GPU with CUDA support.
- Minimum **8GB VRAM** to run the Qwen2-VL 2B model (with 4-bit quantization). **16GB-24GB VRAM** is recommended for larger batch sizes or full datasets.
- **Python:** ```>= 3.9```


## Environment Setup

- Create and activate a Virtual Environment

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

```

- Install PyTorch (Version ```>= 2.6.0``` is strictly required to patch the HuggingFace ```torch.load``` vulnerability)

```bash
pip install --upgrade "torch>=2.6.0" torchvision torchaudio
```
- Install core dependencies
```bash
pip install transformers datasets peft accelerate bitsandbytes qwen-vl-utils faiss-cpu scikit-learn pillow numpy
```

## Usage
The experimental process is divided into independent execution scripts to easily manage results and avoid RAM overflow.

### 1. Run Main System Simulation (Qwen2-VL 2B)
This script automatically initializes the Shared-Engine architecture, builds the FAISS Vector Database, and sequentially evaluates 3 scenarios: Centralized+RAG, Fed-SLM (No RAG), and Proposed (Fed+RAG).
```bash
python main_federated.py
```
You can then input different configs during run time

### 2. Baselines
This project provided seperate scripts for comparison purpose
- Calculate loss based on Epochs
```bash
python calculate_loss.py
```
- Calculate performance based on number of retriever c
```bash
python main_federated_test_retriever_c.py
```
- Centralized setup
```bash
python main_federated_centralized.py
```
- No RAG setup
```bash
python main_federated_no_RAG.py
```
- LLM model setup
```bash
python main_federated_LLM.py
```
- Computational comparison
```bash
python computational_compare.py
```

## Evaluation Metrics
Upon completion, all results are securely saved at: ./data folder. The system is evaluated across 4 standard medical metrics:

- **Closed-ended questions (Yes/No):**
    - **Accuracy:** The absolute diagnostic correctness rate.
    - **F1-Score:** The harmonic mean of Precision and Recall.
- **Open-ended questions (Descriptive):**
    - **BLEU:** Lexical similarity against the doctor's ground truth.
    - **ROUGE-L:** Structural sentence similarity and longest common subsequence matching.
    
## Author

- [Trần Minh Quân](https://github.com/tmquan2002): Writing and Coding
- [Phạm Quang Vinh](https://github.com/vinh1988): Reviewer and Supporter
- [Lê Quang Hùng](https://scholar.google.com/citations?hl=en&user=OivKl1gAAAAJ&view_op=list_works&sortby=pubdate): Instructor and Researcher


## License
This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/). It is suitable for scientific research and academic development purposes.

## Paper
Under work in progress