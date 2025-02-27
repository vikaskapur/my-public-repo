import os

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch


class GemmaFinetuningForFcTest:

    test_prompt = """<bos><start_of_turn>human
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags.You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.Here are the available tools:<tools> [{'type': 'function', 'function': {'name': 'convert_currency', 'description': 'Convert from one currency to another', 'parameters': {'type': 'object', 'properties': {'amount': {'type': 'number', 'description': 'The amount to convert'}, 'from_currency': {'type': 'string', 'description': 'The currency to convert from'}, 'to_currency': {'type': 'string', 'description': 'The currency to convert to'}}, 'required': ['amount', 'from_currency', 'to_currency']}}}, {'type': 'function', 'function': {'name': 'calculate_distance', 'description': 'Calculate the distance between two locations', 'parameters': {'type': 'object', 'properties': {'start_location': {'type': 'string', 'description': 'The starting location'}, 'end_location': {'type': 'string', 'description': 'The ending location'}}, 'required': ['start_location', 'end_location']}}}] </tools>Use the following pydantic model json schema for each tool call you will make: {'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{tool_call}
</tool_call>Also, before making a call to a function take the time to plan the function to take. Make that thinking process between <think>{your thoughts}</think>

Hi, I need to convert 500 USD to Euros. Can you help me with that?<end_of_turn><eos>
<start_of_turn>model
<think>"""

    def __init__(self, is_kaggle_run=False):
        """Initialize the test class"""

        # Load huggingface settings
        if is_kaggle_run:
            from kaggle_secrets import UserSecretsClient

            user_secrets = UserSecretsClient()
            self.hf_model_id = user_secrets.get_secret("HF_MODEL_ID")
            self.hf_username = user_secrets.get_secret("HF_USERNAME")
            os.environ["HF_TOKEN"] = user_secrets.get_secret("HF_TOKEN")
        else:
            from dotenv import load_dotenv

            load_dotenv()

            self.hf_username = os.environ.get("HF_USERNAME")
            self.hf_model_id = os.environ.get("HF_MODEL_ID")

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        # 1. Load the adapter from the hub.
        # 2. Load the base model : "google/gemma-2-2b-it" from the hub.
        # 3. Resize the model with new tokens.

        peft_model_id = f"{self.hf_username}/{self.hf_model_id}"
        config = PeftConfig.from_pretrained(peft_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, peft_model_id)
        model.to(torch.bfloat16)
        model.eval()
        return model, tokenizer

    def run_test(self):
        """Run the test"""
        model, tokenizer = self._load_model_and_tokenizer()
        prompt = self.test_prompt

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,  # Adapt as necessary
            do_sample=True,
            top_p=0.95,
            temperature=0.01,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

        print(f"Result::: \n{tokenizer.decode(outputs[0])}")
        return tokenizer.decode(outputs[0])


if __name__ == "__main__":
    test = GemmaFinetuningForFcTest()
    test.run_test()
