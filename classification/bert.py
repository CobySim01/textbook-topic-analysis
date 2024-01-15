import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from transformers import DistilBertModel, DistilBertTokenizer

import torch

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)


def process_section(data):
    """Function to get BERT embeddings for concatenated concepts and text."""

    combined_text = "[CLS] " + " ".join(data["concepts"]) + " [SEP] " + data["content"]
    inputs = tokenizer(
        combined_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length",
    ).to("cpu")

    with torch.no_grad():
        outputs = model(**inputs.to(model.device))

    embeddings = outputs.last_hidden_state.mean(1).to("cpu")
    embeddings_reshaped = torch.reshape(embeddings, (-1,)).tolist()

    return {"x": embeddings_reshaped, "y": data["topic"]}


def run_bert(integrated_textbook, file=None):
    """Generate all BERT embeddings for an integrated textbook."""
    vectors = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_section, data)
            for data in integrated_textbook.dataset
        ]
        for future in as_completed(futures):
            vectors.append(future.result())

    if file is not None:
        with open(file, "w", encoding="utf-8") as f:
            json.dump(vectors, f)
