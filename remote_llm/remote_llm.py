"""
Customized LLM for support-ai
"""

import argparse
import logging
import os
from typing import Dict, List, Optional
import torch
import yaml

from kserve import Model, ModelServer
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer


CONFIG_INFERENCE_MODEL_PATH = 'inference_model_path'

class RemoteLlamaModel(Model):
    """
    A KServe model wrapper for the Llama causal language model and a sentence
    transformer embeddings model.
    """

    def __init__(self, config):
        """
        Initializes RemoteLlamaModel by loading the specified configuration.
        
        Args:
            config: Configuration dictionary with model paths and 
                    settings.
        
        Raises:
            ValueError: If CONFIG_INFERENCE_MODEL_PATH is missing in config.
            Exception: If model loading fails.
        """
        super().__init__('llama-model')
        self.load(config)

    def load(self, config):
        """
        Loads the Llama tokenizer, inference model, and sentence embeddings 
        model.
        
        Args:
            config: Configuration dictionary with model paths and 
                    settings.
        
        Raises:
            ValueError: If CONFIG_INFERENCE_MODEL_PATH is missing in config.
            Exception: If model loading fails.
        """
        if CONFIG_INFERENCE_MODEL_PATH not in config:
            raise ValueError(f'The config doesn\'t contain {CONFIG_INFERENCE_MODEL_PATH}')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.token = os.getenv('TOKEN', '')
        try:
            logging.info("Loading Llama tokenizer and model...")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                config[CONFIG_INFERENCE_MODEL_PATH], token=self.token
            )
            self.inference_model = LlamaForCausalLM.from_pretrained(
                config[CONFIG_INFERENCE_MODEL_PATH], token=self.token,
                device_map='auto', load_in_4bit=True
            )

            logging.info("Loading Sentence Transformer embeddings model...")
            self.embeddings_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
            self.ready = True
        except Exception as e:
            logging.error("Failed to load models: %s", str(e))
            raise

    async def predict(self, payload: Dict[str, List[str]],
                      _headers: Optional[Dict[str, str]] = None) -> Dict:
        """
        Handles prediction requests, performing inference or embeddings 
        generation based on request type.
        
        Args:
            payload: Contains 'texts' (List[str]) for inference or 
                     embedding and 'type' to specify the operation.
            headers: Optional headers for the request.
        
        Returns:
            dict: Contains the result 'outputs' or an error message.
        """
        texts = payload.get('texts', [])
        response_type = payload.get('type', 'unknown')
        outputs = []

        if not texts:
            return {'error': 'No texts provided for prediction.'}

        try:
            if response_type == 'inference':
                outputs = self._perform_inference(texts)
            elif response_type == 'embeddings':
                outputs = self.__generate_embeddings(texts)
            else:
                return {'error': f'Unknown request type: {response_type}'}
        except Exception as e: # pylint: disable=broad-except
            logging.exception(e)
            return {'error': 'Prediction failed due to internal error.'}

        return {'outputs': outputs}

    def _perform_inference(self, texts: List[str]) -> List[str]:
        """
        Performs text generation using the Llama model.
        
        Args:
            texts: List of input texts to generate responses for.
        
        Returns:
            List[str]: Generated responses for each input text.
        """
        results = []
        for text in texts:
            input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
            generated_ids = self.inference_model.generate(
                input_ids, max_new_tokens=8192, temperature=1, do_sample=False
            )
            output = self.tokenizer.batch_decode(
                generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True
            )[0]
            results.append(output)
        return results

    def __generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for input texts using the SentenceTransformer 
        model.
        
        Args:
            texts: List of input texts to generate embeddings for.
        
        Returns:
            List[List[float]]: Generated embeddings for each input text.
        """
        return [self.embeddings_model.encode(text).tolist() for text in texts]

def get_model_config(path):
    """
    Reads and returns the model configuration from a YAML file.
    
    Args:
        path: Path to the YAML configuration file.
    
    Returns:
        dict: Loaded configuration data.
    """
    config = None
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    return config

def parse_args():
    """
    Parses command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments with model config path.
    """
    parser = argparse.ArgumentParser(description='remote-llm')
    parser.add_argument('--model_config', type=str, default='config.yaml', help='Config path')
    return parser.parse_args()

def main():
    """
    Initializes logging, loads model configuration, and starts the model 
    server.
    """
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = get_model_config(args.model_config)
    model = RemoteLlamaModel(config)
    model_server = ModelServer(http_port=8080, workers=1)
    model_server.start([model])

if __name__ == "__main__":
    main()
