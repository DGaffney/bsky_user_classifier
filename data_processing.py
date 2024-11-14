# data_processing.py
import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
import numpy as np

def load_data(file_path):
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    outdata = [
        {
            "did": e["user_id"],
            "description": e["description"],
            "label_weights": e["user_categories"]
        }
        for e in dataset
        if e["description"] and e["user_categories"]
    ]
    return outdata

def prepare_labels(outdata):
    all_labels = sorted({label for record in outdata for label in record['label_weights'].keys()})
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    y_matrix = np.zeros((len(outdata), len(all_labels)), dtype=float)
    for idx, record in enumerate(outdata):
        for label, weight in record['label_weights'].items():
            y_matrix[idx, label2id[label]] = weight
    return y_matrix, label2id, id2label

class EmbeddingGenerator:
    def __init__(self, model_name='distilbert-base-uncased', device=None):
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.embedding_model = DistilBertModel.from_pretrained(model_name)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(self.device)

    def generate_embeddings(self, descriptions, batch_size=1000):
        all_embeddings = []
        descriptions = [desc for desc in descriptions]
        for i in range(0, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i + batch_size]
            inputs = self.tokenizer(
                batch_descriptions,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
        return np.vstack(all_embeddings)
