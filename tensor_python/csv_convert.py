import tensorflow as tf

model = tf.keras.models.load_model('./models/petals2')

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('./models/petals.tflite', 'wb') as f:
    f.write(tflite_model)

# reinitialized_model = keras.models.model_from_json(json_config)
# with open('./models/petals.json', 'w') as f:
#  f.write(model.to_json())


def predict(expected, challenge):
    respuesta = model.predict([challenge])[0]
    max_item = max(respuesta)
    index_list = [index for index in range(
        len(respuesta)) if respuesta[index] == max_item]
    print("Expected:"+str(expected)+" Predicted:"+str(index_list))

predict(0, [5.0, 3.2, 1.2, 0.2])
predict(1, [6.0, 2.7, 5.1, 1.6])
predict(2, [6.4, 3.1, 5.5, 1.8])

predict(0, [5.4, 3.7, 1.5, 0.2])
predict(1, [5.9, 3.2, 4.8, 1.8])
predict(2, [7.2, 3.6, 6.1, 2.5])
