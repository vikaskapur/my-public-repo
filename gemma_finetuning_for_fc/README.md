# Gemma Fine-tuning for Function Calling

This repository contains code and resources for fine-tuning the Gemma model for a specific use case in the FC domain.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to fine-tune the Gemma model to improve its performance for for function calling. The repository includes scripts for model training, and testing.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/vikaskapur/my-public-repo.git
cd my-public-repo/gemma_finetuning_for_fc
python -m venv exp-env
source exp-env/bin/activate
pip install -r requirements.txt
```

## Usage

2. **Model Training**: Train the model and upload it on hugging face.
    ```bash
    python train.py 
    ```

3. **Test**: Test the model.
    ```bash
    python test.py 
    ```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.