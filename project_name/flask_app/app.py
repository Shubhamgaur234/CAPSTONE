from flask import Flask, render_template, request
import mlflow
import pickle
import pandas as pd
import numpy as np
import time
import re
import string
import warnings
import dagshub

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST
)

warnings.filterwarnings("ignore")


# -----------------------------
# TEXT PREPROCESSING FUNCTIONS
# -----------------------------

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)


def removing_numbers(text):
    return ''.join(
        char for char in text
        if not char.isdigit()
    )


def lower_case(text):
    return text.lower()


def removing_punctuations(text):
    text = re.sub(
        '[%s]' % re.escape(string.punctuation),
        ' ',
        text
    )
    text = re.sub(
        '\s+',
        ' ',
        text
    ).strip()
    return text


def removing_urls(text):
    pattern = re.compile(
        r'https?://\S+|www\.\S+'
    )
    return pattern.sub('', text)


def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text


# -----------------------------
# DAGSHUB / MLFLOW
# -----------------------------

mlflow.set_tracking_uri(
    "https://dagshub.com/gaur3786/CAPSTONE.mlflow"
)

dagshub.init(
    repo_owner="gaur3786",
    repo_name="CAPSTONE",
    mlflow=True
)


# -----------------------------
# FLASK APP
# -----------------------------

app = Flask(__name__)


# -----------------------------
# PROMETHEUS METRICS
# -----------------------------

registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total requests",
    ["method", "endpoint"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency",
    ["endpoint"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction count",
    ["prediction"],
    registry=registry
)


# -----------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------

# Your registered model
model_uri = "models:/my_model/4"

print(
    f"Loading model from {model_uri}"
)

model = mlflow.pyfunc.load_model(
    model_uri
)

vectorizer = pickle.load(
    open(
        "models/vectorizer.pkl",
        "rb"
    )
)


# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():

    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/"
    ).inc()

    start_time = time.time()

    response = render_template(
        "index.html",
        result=None
    )

    REQUEST_LATENCY.labels(
        endpoint="/"
    ).observe(
        time.time() - start_time
    )

    return response


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict"
    ).inc()

    start_time = time.time()

    text = request.form["text"]

    cleaned_text = normalize_text(
        text
    )

    features = vectorizer.transform(
        [cleaned_text]
    )

    features_df = pd.DataFrame(
        features.toarray(),
        columns=[
            str(i)
            for i in range(
                features.shape[1]
            )
        ]
    )

    result = model.predict(
        features_df
    )

    prediction = result[0]

    PREDICTION_COUNT.labels(
        prediction=str(prediction)
    ).inc()

    REQUEST_LATENCY.labels(
        endpoint="/predict"
    ).observe(
        time.time() - start_time
    )

    return render_template(
        "index.html",
        result=prediction
    )


@app.route("/metrics")
def metrics():

    return (
        generate_latest(
            registry
        ),
        200,
        {
            "Content-Type":
            CONTENT_TYPE_LATEST
        }
    )


# -----------------------------
# RUN APP
# -----------------------------

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )