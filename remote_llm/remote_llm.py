import logging
import os
import torch

from kserve import Model, ModelServer
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Dict, List, Optional


class RemoteLlamaModel(Model):
    def __init__(self):
        super().__init__('llama-model')
        self.load()

    def load(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.token = os.getenv('TOKEN', '')
        try:
            logging.info("Loading Llama tokenizer and model...")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf", token=self.token
            )
            self.inference_model = LlamaForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf", token=self.token, device_map='auto', load_in_4bit=True
            )

            logging.info("Loading Sentence Transformer embeddings model...")
            self.embeddings_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
            self.ready = True
        except Exception as e:
            logging.error(f"Failed to load models: {str(e)}")
            raise

    async def predict(self, payload: Dict[str, List[str]], headers: Optional[Dict[str, str]] = None) -> Dict:
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
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            return {'error': 'Prediction failed due to internal error.'}

        return {'outputs': outputs}

    def _perform_inference(self, texts: List[str]) -> List[str]:
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
        return [self.embeddings_model.encode(text).tolist() for text in texts]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = RemoteLlamaModel()
    model_server = ModelServer(http_port=8080, workers=1)
    model_server.start([model])
