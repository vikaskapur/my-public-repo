from enum import Enum
from functools import partial
import pandas as pd
import torch
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType


seed = 42
set_seed(seed)

from dotenv import load_dotenv

load_dotenv()


class FineTuneForFunctionCalling:

    def __init__(self, is_dev_run=False, is_kaggle_run=False):
        """Initialize the FineTuneForFunctionCalling class"""

        # model to fine-tune for function calling
        self.model_name = "google/gemma-2-2b-it"

        # function calling dataset with thinking
        # "NousResearch/hermes-function-calling-v1" is a popular dataset for function calling.
        # It is further enhanced by adding some new **thinking** step computer from deepseek-ai DeepSeek-R1-Distill-Qwen-32B
        self.dataset_name = "Jofthomas/hermes-function-calling-thinking-V1"  # function calling dataset with thinking

        # Load huggingface settings
        if is_kaggle_run:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            self.hf_model_id = user_secrets.get_secret("HF_MODEL_ID")
            self.hf_username = user_secrets.get_secret("HF_USERNAME")
        else:
            self.hf_username = os.environ.get("HF_USERNAME")
            self.hf_model_id = os.environ.get("HF_MODEL_ID")

        # Is this a dev run?
        self.is_dev_run = is_dev_run

    def fine_tune(self):
        """Fine-tune the model for function calling"""

        # Simplify the chat template
        self._simplify_chat_template()

        # Prepare dataset for training
        dataset = self._prepare_dataset()

        # Print sample dataset
        self._print_dataset_example()

        # Modify the tokenizer
        self._modify_tokenizer()

        # load the model
        self._load_model()

        # Configure LoRA parameters
        peft_config = self._configure_lora()

        # Set training arguments
        training_arguments = self._set_training_arguments()

        # Train the model
        trainer = self._train_model(dataset, peft_config, training_arguments)

        # Save model to huggingface hub
        self._save_model_to_hub(trainer)

    def _simplify_chat_template(self):
        # Simplified Chat template
        # 1. Remove validation exception for : Conversation roles must alternate user/assistant/user/assistant/...
        # 2. Remove change role "assistant" --> "model"
        # 3. Add <eos> tag at with <end_of_turn>

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        print("+++++++ Model Original Chat Template ++++++++++++")
        print(self.tokenizer.chat_template)
        print("------- Model Original Chat Template ------------")

        self.tokenizer.chat_template = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn><eos>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"

        print("+++++++ Model New Chat Template ++++++++++++")
        print(self.tokenizer.chat_template)
        print("------- Model New Chat Template ------------")

    def _prepare_dataset(self):
        """Prepare the dataset for training"""
        if self.is_dev_run:
            split = "train[0:10]"  # reduce the working size to speed up iteration
            dataset["train"] = load_dataset(self.dataset_name, split=split)
        else:
            dataset = load_dataset(self.dataset_name)
        dataset = dataset.rename_column("conversations", "messages")
        dataset = dataset.map(self._preprocess, remove_columns="messages")
        dataset = dataset["train"].train_test_split(0.1)
        print(f"Dataset stats: \n{dataset}")
        return dataset

    # pre-process list of messages, to a prompt that the model can understand.
    def _preprocess(self, sample):
        messages = sample["messages"]
        first_message = messages[0]

        # Instead of adding a system message, we merge the content into the first user message
        if first_message["role"] == "system":
            system_message_content = first_message["content"]
            # Merge system content with the first user message
            messages[1]["content"] = (
                system_message_content
                + "Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>\n\n"
                + messages[1]["content"]
            )
            # Remove the system message from the conversation
            messages.pop(0)

        return {"text": self.tokenizer.apply_chat_template(messages, tokenize=False)}

    def _print_dataset_example(self):
        """Print the dataset example"""
        # Let's look at how we formatted the dataset

        # In this example we have :
        # 1. A *User message* containing the **necessary information with the list of available tools** inbetween `<tools></tools>` then the user query, here:  `"Can you get me the latest news headlines for the United States?"`

        # 2. An *Assistant message* here called "model" to fit the criterias from gemma models containing two new phases, a **"thinking"** phase contained in `<think></think>` and an **"Act"** phase contained in `<tool_call></<tool_call>`.

        # 3. If the model contains a `<tools_call>`, we will append the result of this action in a new **"Tool"** message containing a `<tool_response></tool_response>` with the answer from the tool.

        print(self.dataset["train"][7]["text"])

    def _modify_tokenizer(self):
        """Modify the tokenizer"""
        # The tokenizer splits text into sub-words by default. This is **not** what we want for our new special tokens!
        # While we segmented our example using `<think>`, `<tool_call>`, and `<tool_response>`, the tokenizer does **not** yet treat them as whole tokens—it still tries to break them down into smaller pieces. To ensure the model correctly interprets our new format, we must **add these tokens** to our tokenizer.

        # Additionally, since we changed the `chat_template` in our **preprocess** function to format conversations as messages within a prompt, we also need to modify the `chat_template` in the tokenizer to reflect these changes.

        # Add the new special tokens
        class ChatmlSpecialTokens(str, Enum):
            tools = "<tools>"
            eotools = "</tools>"
            think = "<think>"
            eothink = "</think>"
            tool_call = "<tool_call>"
            eotool_call = "</tool_call>"
            tool_response = "<tool_response>"
            eotool_response = "</tool_response>"
            pad_token = "<pad>"
            eos_token = "<eos>"

            @classmethod
            def list(cls):
                return [c.value for c in cls]

        simplified_chat_template = self.tokenizer.chat_template

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            pad_token=ChatmlSpecialTokens.pad_token.value,
            additional_special_tokens=ChatmlSpecialTokens.list(),
        )
        self.tokenizer.chat_template = simplified_chat_template

    def _load_model(self):
        """Load the model"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="eager",  # recommended for gemma models
            device_map="auto",
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(torch.bfloat16)

    def _configure_lora(self):
        """Configure the LoRA"""
        # Configure LoRA parameters

        # r: rank dimension for LoRA update matrices (smaller = more compression)
        rank_dimension = 16
        # lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
        lora_alpha = 64

        # lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
        lora_dropout = 0.05

        peft_config = LoraConfig(
            r=rank_dimension,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "gate_proj",
                "q_proj",
                "lm_head",
                "o_proj",
                "k_proj",
                "embed_tokens",
                "down_proj",
                "up_proj",
                "v_proj",
            ],  # wich layer in the transformers do we target
            task_type=TaskType.CAUSAL_LM,
        )

        return peft_config

    def _set_training_arguments(self):
        """Set the training arguments"""

        output_dir = (
            self.hf_model_id
        )  # The directory where the trained model checkpoints, logs, and other artifacts will be saved. It will also be the default name of the model when pushed to the hub if not redefined later.
        per_device_train_batch_size = 1  # batch size per GPU
        per_device_eval_batch_size = 1  # batch size per GPU/
        gradient_accumulation_steps = 1  # 4, Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        logging_steps = 1  # 5, Number of update steps between two logs.
        learning_rate = 1e-4  # The initial learning rate for the optimizer.

        max_grad_norm = 1.0  # Maximum gradient norm (for gradient clipping) default = 1
        num_train_epochs = 1
        warmup_ratio = 0.1  # Ratio of total training steps used for a linear warmup from 0 to learning_rate
        lr_scheduler_type = "cosine"  # The scheduler type to use
        max_seq_length = (
            1500  #  maximum sequence length to use for the ConstantLengthDataset
        )

        training_arguments = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            save_strategy="no",  # checkpoint save strategy to adopt during training
            eval_strategy="epoch",
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            weight_decay=0.1,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            report_to="tensorboard",
            bf16=True,
            hub_private_repo=False,  # Whether to make the repo private
            push_to_hub=False,  # Whether or not to push the model to the Hub every time the model is saved
            num_train_epochs=num_train_epochs,
            gradient_checkpointing=True,  # If True, use gradient checkpointing to save memory at the expense of slower backward pass.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            # enabling packing, results change in number of steps
            packing=False,  # multiple short examples are packed in the same input sequence to increase training efficiency.
            max_seq_length=max_seq_length,
        )

        return training_arguments

    def _train_model(self, dataset, peft_config, training_arguments):
        """Train the model"""

        # As Trainer, we use the `SFTTrainer` which is a Supervised Fine-Tuning Trainer.
        trainer = SFTTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )

        # Here, we launch the training 🔥. Perfect time for you to pause and grab a coffee ☕."""
        trainer.train()
        trainer.save_model()  # Will save the model, so you can reload it using from_pretrained()
        return trainer

    def _save_model_to_hub(self, trainer):
        """Push the Model and the Tokenizer to the Hub"""

        username = self.hf_username
        output_dir = self.hf_model_id

        trainer.push_to_hub(f"{username}/{output_dir}")
        # Since we also modified the **chat_template** Which is contained in the tokenizer, let's also push the tokenizer with the model.
        # self.tokenizer.eos_token = "<eos>"
        # push the tokenizer to hub ( replace with your username and your previously specified
        self.tokenizer.push_to_hub(f"{username}/{output_dir}", token=True)


if __name__ == "__main__":
    """Run the fine-tuning for function calling"""

    fine_tune = FineTuneForFunctionCalling(is_dev_run=True, is_kaggle_run=True)
    fine_tune.fine_tune()
