# train.py
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.stats import rankdata
import joblib
import json

from data_processing import load_data, EmbeddingGenerator, prepare_labels
from utils import compute_ndcg

def main():
    # Load data
    outdata = load_data("labeled_users.json")

    # Extract descriptions
    descriptions = [record['description'] for record in outdata]

    # Generate embeddings
    embedder = EmbeddingGenerator()
    X_embeddings = embedder.generate_embeddings(descriptions)

    # Prepare labels
    y_matrix, label2id, id2label = prepare_labels(outdata)

    # Save label mappings for later use
    mappings = {'label2id': label2id, 'id2label': id2label}
    with open('label_mappings.json', 'w') as f:
        json.dump(mappings, f)

    # K-Fold Cross Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_ndcgs = []

    # Store embeddings in outdata
    for idx, record in enumerate(outdata):
        record['embedding'] = X_embeddings[idx].tolist()

    for fold, (train_index, test_index) in enumerate(kf.split(X_embeddings)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_train, X_test = X_embeddings[train_index], X_embeddings[test_index]
        y_train, y_test = y_matrix[train_index], y_matrix[test_index]
        # Initialize and train the model
        xgb_base = XGBRegressor(n_estimators=1500, random_state=42)
        multi_xgb = MultiOutputRegressor(xgb_base, n_jobs=-1)
        multi_xgb.fit(X_train, y_train)
        # Predict on the validation set
        y_pred = multi_xgb.predict(X_test)
        # Loop over each sample in the validation set
        for i, pred in enumerate(y_pred):
            actual_weights = y_test[i]
            predicted_weights = pred
            non_zero_indices = np.where(actual_weights != 0)[0]
            if len(non_zero_indices) > 0:
                actual_weights_nonzero = actual_weights[non_zero_indices]
                predicted_weights_nonzero = predicted_weights[non_zero_indices]
                ndcg_score = compute_ndcg(actual_weights_nonzero, predicted_weights_nonzero)
                all_ndcgs.append(ndcg_score)
                # Store predicted weights and ranks in outdata for each item
                test_idx = test_index[i]
                outdata[test_idx]["predicted_weights"] = {
                    id2label[idx]: float(predicted_weights[idx]) for idx in non_zero_indices
                }
                outdata[test_idx]["predicted_ranks"] = {
                    id2label[idx]: int(rank) for idx, rank in zip(
                        non_zero_indices,
                        rankdata(-predicted_weights_nonzero, method='ordinal')
                    )
                }
        print(f"Average NDCG for fold {fold + 1}: {np.mean(all_ndcgs):.4f}")

    # Save enriched data
    with open("enriched_data.json", "w") as f:
        for row in outdata:
            f.write(json.dumps(row) + '\n')

    # Train final model on all data
    xgb_base = XGBRegressor(n_estimators=1500, random_state=42)
    multi_xgb = MultiOutputRegressor(xgb_base, n_jobs=-1)
    multi_xgb.fit(X_embeddings, y_matrix)
    # Save the trained model
    joblib.dump(multi_xgb, 'xgboost_model.joblib')
    print("Model training complete and saved as 'xgboost_model.joblib'.")

if __name__ == "__main__":
    main()
