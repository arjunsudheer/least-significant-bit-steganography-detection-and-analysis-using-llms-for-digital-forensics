# least-significant-bit-steganography-detection-and-analysis-using-large-language-models-for-digital-forensics

This repository is dedicated to our source code for our research paper titled [Least Significant Bit Steganography Detection and Analysis Using Large Language Models for Digital Forensics](https://ieeexplore.ieee.org/abstract/document/11642309). We presented our research work at the Silicon Valley Cybersecurity Conference 2026.

We present an end-to-end framework for detecting and analyzing Least Significant Bit (LSB) steganography in digital images, combining deep learning classification with automated forensic reporting. A Spatial Rich Model (SRM)-based convolutional neural network is trained to distinguish clean images from those carrying hidden payloads embedded across five malicious categories: JavaScript, obfuscated JavaScript in HTML, PowerShell scripts, Ethereum wallet addresses, and URL/IP addresses. Detected stego images are passed to an LSB payload extractor that precisely reverses the RobinDavid embedding algorithm to recover the hidden content. A LangGraph-orchestrated forensics agent then searches academic literature and threat-intelligence databases, classifies the payload type with code-level evidence, and produces a structured PDF report with inline citations. The quality of the generated reports is evaluated using RAGAS faithfulness and answer relevancy metrics against the retrieved evidence, with all inference and evaluation running locally via Ollama.

## Local Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### Set up

```bash
# 1. Clone the repository
git clone https://github.com/arjunsudheer/llm-based-image-steganography-detection.git
cd llm-based-image-steganography-detection
```

Set your Tavily API key in the .env file.

Download the [Stego-Images-Dataset](https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset) from Kaggle and place it at ```./dataset/```. The dataset already contains the train/validation/test split.

### Installation

```bash
# 1. Install Python dependencies in a virtual environment
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# 2. Pull the LLM and embedding model
ollama pull ministral-3:3b
ollama pull nomic-embed-text
```

### Running the pipeline

We have provided a script ```run.sh``` for your convenience. This will train the CNN binary classification model, and generate the reports using the forensics agent.

```bash
chmod +x run.sh
./run.sh
```
