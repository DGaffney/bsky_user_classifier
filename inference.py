# inference.py
import torch
import numpy as np
import joblib
import json
from transformers import DistilBertTokenizerFast, DistilBertModel

class Predictor:
    def __init__(self, model_path='xgboost_model.joblib', mappings_path='label_mappings.json', device=None):
        # Load the XGBoost model
        self.model = joblib.load(model_path)

        # Load label mappings
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        self.id2label = {int(k): v for k, v in mappings['id2label'].items()}

        # Load the tokenizer and embedding model
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        self.embedding_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model.to(self.device)

    def generate_embedding(self, text):
        inputs = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding

    def predict(self, text):
        embedding = self.generate_embedding(text)
        y_pred = self.model.predict(embedding)
        predictions = {self.id2label[i]: float(y_pred[0][i]) for i in range(len(self.id2label))}
        return predictions

# Example usage
if __name__ == "__main__":
    predictor = Predictor()
    text = "I write about American politics"
    predictions = predictor.predict(text)
    print(predictions)
