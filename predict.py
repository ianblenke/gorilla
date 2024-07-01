import os
from cog import BasePredictor, Input
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def get_prompt(user_query: str) -> str:
    """
    Generates a conversation prompt based on the user's query.

    Parameters:
    - user_query (str): The user's query.

    Returns:
    - str: The formatted conversation prompt.
    """
    return f"USER: <<question>> {user_query}\nASSISTANT: "

class Predictor(BasePredictor):
    def setup(self):
        """
        Load the model into memory to make running multiple predictions efficient.
        Sets up the device, model, tokenizer, and pipeline for text generation.
        """
        # Device setup
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Model and tokenizer setup
        model_id = os.environ.get('MODEL', 'gorilla-openfunctions-v2')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True
        )

        # Move model to device
        self.model.to(self.device)

        # Pipeline setup
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

    def predict(self, user_query: str = Input(description="User's query")) -> str:
        """
        Run a single prediction on the model using the provided user query.

        Parameters:
        - user_query (str): The user's query for the model.

        Returns:
        - str: The model's generated text based on the query.
        """
        prompt = get_prompt(user_query)
        output = self.pipe(prompt)
        return output
