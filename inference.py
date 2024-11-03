from tensorflow.keras.models import load_model

model = load_model("Sentiments classifier.keras")

review_example = ''
def make_prediction(sentence):
    tokens = tokenizer(sentence)
    tokens = tf.expand_dims(tokens, 0)
    prediction = fnet_classifier.predict(tokens, verbose=0)

    if prediction[0][0] > 0.5:
        result = "The review is POSITIVE"
    else:
        result = "The review is NEGATIVE"
    return result
result = make_prediction(review_example)
print(result)
