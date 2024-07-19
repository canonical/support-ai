import argparse

from kserve import Model, ModelServer
from sentence_transformers import SentenceTransformer
from transformers import LlamaForCausalLM, LlamaTokenizer
from typing import Dict


class RemoteLlamaModel(Model):
    def __init__(self, token):
        self.name = 'llama-model'
        self.token = token
        super().__init__(self.name)
        self.load()

    def load(self):
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=self.token)
        self.inference_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=self.token)
        self.embeddings_model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

    async def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        texts = payload['texts']
        outputs = []
        match payload['type']:
            case 'inference':
                for text in texts:
                    input_ids = self.tokenizer.encode(text, return_tensors="pt")
                    generated_ids = self.inference_model.generate(input_ids, max_new_tokens=8192, temperature=1, do_sample=False)
                    output = self.tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                    outputs.append(output)
            case 'embeddings':
                for text in texts:
                    embeddings = self.embeddings_model.encode(text)
                    outputs.append(embeddings.tolist())

        return {'outputs': outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Remote model parameters.')
    parser.add_argument('-t', '--token', type=str, help='Huggingface Llama2 access token', dest='token')
    args = parser.parse_args()
    model = RemoteLlamaModel(args.token)
    model_server = ModelServer(http_port=8080, workers=1)
    model_server.start([model])
