import tensorflow as tf

model = tf.keras.models.load_model('./models/digits')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('./models/digits.tflite', 'wb') as f:
  f.write(tflite_model)