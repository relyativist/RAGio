---
title: RAGio
emoji: 🌐
colorFrom: purple
colorTo: cyan
sdk: gradio
sdk_version: 4.27.0
---

# RAGio - simple to start RAG with HuggingFace gradio interface.

Retrieval-Augmented Generation (RAG) with a Gradio interface. Perfect for both beginners and experienced developers looking to integrate advanced NLP features.

## Features

- **Beginner Friendly**: Easily set up and run your RAG models locally or host them on HuggingFace Spaces. Currently supports *.pdf documents only.
- **Interactive UI**: Engage with your models and data in real time through a dynamic Gradio interface.
- **HuggingFace and OpenAI APIs**: Utilizes HuggingFace and OpenAI API.
- **Vector store with LanceDB**: Utilizes LanceDB to store and manage embedding vectors.
- **Multiple Chunking Strategies**: Coming soon ...  

## Quick Start

Get started with RAGio by cloning the repository and installing dependencies:

```bash
git clone https://github.com/your-username/RAGio.git
cd RAGio
pip install -r requirements.txt
```

### Run the application

**Important!**\
Ensure to configure your environment by filling in the *.template.env* file with your HuggingFace and OpenAI credentials. Rename this file to *.env* after updating.

```bash
cd RAGio
source ./.env  # apply environment variables
gradio app.py  # run gradio app
```

Open http://127.0.0.1:7860 in your browser.

For more details on configuration and usage, check out our documentation.

## Contributing

Contributing
Your contributions are welcome! If you have suggestions or want to improve RAGio, feel free to fork the repository, make changes, and submit a pull request.
