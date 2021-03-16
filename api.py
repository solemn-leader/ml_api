import os
from urllib.parse import unquote
from transformers import DistilBertTokenizer

import flask
from flask import request

port = int(os.environ.get("PORT", 5000))

app = flask.Flask(__name__)
app.config["DEBUG"] = True
model = InferenceSession('onnx_models/model-quantized.onnx', providers=["CPUExecutionProvider"])
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

label_encoder_classes = ['Government News', 'Middle-east', 'News', 'US_News', 'left-news', 'politics']

def get_preds(text):
    model_inputs = tokenizer(
        text, return_tensors='pt', truncation=True,
        padding=True
    )
    inputs_onnx = {k: v.cpu().detach().numpy() for k, v in model_inputs.items()}
    outp = model.run(None, inputs_onnx)[0][0]
    class_ind = outp.argmax()
    return label_encoder_classes[class_ind]


@app.route('/', methods=['GET'])
def home():
    return '<h1>Index , Baby</h1>'

@app.route('/api', methods=['GET'])
def api():
    text = unquote(str(request.query_string))[2:-1]
    return f'<h1>%s</h1>' % text


@app.route('/ml', methods=['GET'])
def ml():
    text = unquote(str(request.query_string))[2:-1]
    return get_preds(text)

@app.route('/test', methods=['GET'])
def test():
    return "test"

app.run(host='0.0.0.0', port=port)
