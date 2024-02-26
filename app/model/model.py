from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import torch.nn.functional as F
from pathlib import Path
import torch
import os


BASE_DIR = Path(__file__).resolve(strict=True).parent

def download_model():

    model_path = str(BASE_DIR) + "/model_artifacts"

    if not os.path.exists(model_path):
        # Create the directory
        os.makedirs(model_path)

    
    model_id = 'BAAI/bge-large-en-v1.5'
    model = ORTModelForFeatureExtraction.from_pretrained("codefactory4791/quantized-bge-large-en-v1.5")
    tokenizer = AutoTokenizer.from_pretrained(model_id)


    # comment this later to check if not saving the model objects speed up embedding generation
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return model, tokenizer

model, tokenizer = download_model()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    
def generate_embeddings(query):

    encoded_inputs = tokenizer(query, padding = True, truncation = True, return_tensors = 'pt')

    attention_mask = encoded_inputs['attention_mask']

    outputs = model(**encoded_inputs)

    sentence_embeddings = mean_pooling(outputs, attention_mask)

    sentence_embeddings = F.normalize(sentence_embeddings, p = 2, dim = 1)

    return sentence_embeddings.tolist()

