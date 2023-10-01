from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Specify the model name or path
# model_name = "/home/motua16/Documents/Projects/mlops/noahgiftbook/chap4a/"
model_name = './webapp/'

import os
print(os.getcwd())


# Load the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.eval()  # Set the model to evaluation mode

# Load the tokenizer with padding enabled
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

@app.route("/", methods=["GET"])
def home():
    return "Welcome"

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json[0]

    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        # max_length=128,  # You can adjust the maximum sequence length as needed
    )

    # Perform model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the predicted class (0 or 1)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted class to a boolean (positive/negative)
    result = bool(predicted_class)

    return jsonify({"positive": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
