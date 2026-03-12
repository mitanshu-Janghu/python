import tensorflow as tf
from tensorflow import keras

# model load
model = keras.models.model_from_json(open("config.json").read())
model.load_weights("model.weights.h5")

# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save
open("emotion_model.tflite", "wb").write(tflite_model)