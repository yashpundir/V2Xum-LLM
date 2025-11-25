# V2Xum-LLM: Cross-Modal Video Summarization with Temporal Prompt Instruction Tuning (AAAI 2025)
[**🌐 Homepage**](https://hanghuacs.github.io/v2xum/) | [**🔬 Paper**](https://arxiv.org/pdf/2404.12353.pdf) | [**👩‍💻 Code**](https://github.com/hanghuacs/V2Xum-LLM) | [**📊 Dataset**](https://github.com/hanghuacs/v2xum_dataset)

---

## Our Project: Fine-tuning V2Xum-LLM on ActivityNet for NLP

This repository extends the V2Xum-LLM framework and paper for an NLP-focused project. While we did not introduce major changes to the original methodology, our main contribution is the fine-tuning and evaluation of large language models on video data.

**What we did:**
- We fine-tuned the Vicuna (Llama) model on approximately 1,000 videos from the ActivityNet dataset using the V2Xum-LLM methodology.
- The model was subsequently evaluated on an additional 2,000 videos from ActivityNet.
- Our focus was on cross-modal video summarization tasks, leveraging temporal prompt instruction tuning from the original paper.

**Results:**
- On the test set, we observed strong performance in video summarization as measured by Fclip and Cross Fclip scores.
- This suggests that temporal prompt-tuning with ActivityNet video data and the Vicuna model is effective for NLP/video summarization tasks.

**Contributors:**  
- Siddharth (t25103)  
- Mohit (s25019)  
- Yashasvi (s25018)  
- Sahil Shukla (d25044)  
- Navdeep (s25009)  
- Sanjay (ptd2508)  

---

## Usage

### Preparation

```bash
conda create -n v2xumllm python=3.9 -y
```

```bash
conda activate v2xumllm
```

### Demo

```
python -m v2xumllm.inference
```

## ✏️ Citation
```bibtex
@article{hua2024v2xum,
  title={V2xum-llm: Cross-modal video summarization with temporal prompt instruction tuning},
  author={Hua, Hang and Tang, Yunlong and Xu, Chenliang and Luo, Jiebo},
  journal={arXiv preprint arXiv:2404.12353},
  year={2024}
}
```