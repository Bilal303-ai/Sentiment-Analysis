import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import keras_nlp

fnet_classifier = load_model("Sentiments classifier.keras")

review_example = input("Input your review: ")

with open("vocab.json", "r") as f:
    vocab = json.load(f)

seq_max_length = 512
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=vocab,
    lowercase=False,
    sequence_length=seq_max_length,
)

def make_prediction(sentence):
    tokens = tokenizer(review_example)
    tokens = tf.expand_dims(tokens, 0)
    prediction = fnet_classifier.predict(tokens, verbose=0)

    if prediction[0][0] > 0.5:
        result = "The review is POSITIVE"
    else:
        result = "The review is NEGATIVE"
    return result
result = make_prediction(review_example)
print(result)
