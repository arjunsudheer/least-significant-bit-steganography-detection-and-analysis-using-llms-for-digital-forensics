# LLM-Based Image Steganography Detection

This repository presents an end-to-end framework for detecting and analysing Least Significant Bit (LSB) steganography in digital images, combining deep learning classification with automated forensic reporting. A Spatial Rich Model (SRM)-based convolutional neural network is trained to distinguish clean images from those carrying hidden payloads embedded across five malicious categories: JavaScript, obfuscated JavaScript in HTML, PowerShell scripts, Ethereum wallet addresses, and URL/IP addresses. Detected stego images are passed to an LSB payload extractor that precisely reverses the RobinDavid embedding algorithm to recover the hidden content. A LangGraph-orchestrated forensics agent then searches academic literature and threat-intelligence databases, classifies the payload type with code-level evidence, and produces a structured PDF report with inline citations and SHAP attribution maps. The quality of the generated reports is evaluated using RAGAS faithfulness and answer relevancy metrics against the retrieved evidence, with all inference and evaluation running locally via Ollama.

## Local Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (8 GB+ VRAM recommended; CPU inference is supported but slow)
- [Ollama](https://ollama.com) installed and running

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/arjunsudheer/llm-based-image-steganography-detection.git
cd llm-based-image-steganography-detection

# 2. Install Python dependencies in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# 3. Pull the LLM and embedding model
ollama pull ministral-3:3b
ollama pull nomic-embed-text
```

Set your Tavily API key in the .env file.

Download the [Stego-Images-Dataset](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset) from Kaggle and place it at ```./dataset/```.

### Running the pipeline

We have provided a script ```run.sh``` for your convenience. This will train the CNN binary classification model, and generate the reports using the forensics agent.

```bash
chmod +x run.sh
./run.sh
```
