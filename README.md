# My LLM & AI Agents Lab

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-red.svg)](https://huggingface.co/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An experimental playground for exploring and documenting recent advances in Large Language Models (LLMs) and AI agents. This repository contains a series of experiments, code snippets, and detailed documentation—all implemented in Python using Hugging Face and PyTorch.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Code Quality](#code-quality)
- [License](#license)
- [Contact](#contact)

---

## About the Project

This project serves as my personal lab for experimenting with state-of-the-art LLMs and AI agent frameworks. Here, you’ll find:

- **Innovative Experiments:** Cutting-edge experiments exploring the behavior and performance of LLMs.
- **In-depth Documentation:** Step-by-step guides, observations, and analyses of various AI agent architectures.
- **Hands-On Code:** Implementations mostly in Python, with extensive use of Hugging Face libraries and PyTorch to push the boundaries of current AI research.

The goal is to both document my learning journey and provide a resource for others interested in building and understanding modern AI agents.

---

## Features

- **Modular Experiments:** Easily switch between different LLM and agent experiments.
- **Comprehensive Documentation:** Detailed notes and guides accompany every experiment.
- **Cutting-edge Tools:** Leveraging Hugging Face’s Transformers, PyTorch, and other modern libraries.
- **Interactive Examples:** Code snippets that can be directly run and modified to explore new ideas.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vikaskapur/my-public-repo.git
   cd my-public-repo/<exp>
   ```

2. **Create an environment and install dependencies:**

 - Mac/Linux/WSL
```
$ python3 -m venv exp-env
$ source exp-env/bin/activate
$ pip install -r requirements.txt
```
- Windows Powershell
```
PS> python3 -m venv exp-env
PS> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
PS> expv\scripts\activate
PS> pip install -r requirements.txt
```

---

## Usage

Explore the experiments by running the Python scripts located in the experiments directory. 

For example:
```
python test.py
```

You can also find detailed instructions in the README for each experiment.

---

## Project Structure

```
my-public-repo/
|
|── experiment/         # Experiment scripts and notebooks
│   ├── experiment_llm.py
│   └── requirements.txt
├── LICENSE              # License file
└── README.md            # This file
```

---

## Code Quality


- Follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Ruff for formatting and linting checks
- Mypy + Pyright for typing 1, 2
- Black for formatting
- isort for import sorting

---

## License

Distributed under the MIT License. See LICENSE for more information.

---

## Contact

Vikas Kapur – vikaskapur04@gmail.com





