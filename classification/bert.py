from concurrent.futures import ProcessPoolExecutor, as_completed

import tensorflow as tf
import torch
from transformers import DistilBertModel, DistilBertTokenizer

tf.experimental.numpy.experimental_enable_numpy_behavior()

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)


def process_section(data, with_concepts):
    """Function to get BERT embeddings for concatenated concepts and text."""
    if with_concepts:
        text = "[CLS] " + " ".join(data["concepts"]) + " [SEP] " + data["content"]
    else:
        text = "[CLS] " + data["content"]

    inputs = tokenizer(
        text,
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


def run_bert(dataset, with_concepts):
    """Generate all BERT embeddings for an integrated textbook."""
    vectors = []
    with ProcessPoolExecutor() as executor:
        futures = (
            executor.submit(process_section, data, with_concepts) for data in dataset
        )
        for future in as_completed(futures):
            vectors.append(future.result())

    X = tf.convert_to_tensor([v["x"] for v in vectors])
    y = tf.convert_to_tensor([v["y"] for v in vectors])
    return X, y
