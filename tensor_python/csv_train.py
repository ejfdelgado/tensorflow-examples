import tensorflow as tf
import pandas as pd

# https://rubikscode.net/2021/08/03/introduction-to-tensorflow-with-python-example/

COLUMN_NAMES = [
        'SepalLength', 
        'SepalWidth',
        'PetalLength', 
        'PetalWidth', 
        'Species'
        ]

# Import training dataset
training_dataset = pd.read_csv('./datasets/iris/iris_training.csv', names=COLUMN_NAMES, header=0, delimiter=";")
train_x = training_dataset.iloc[:, 0:4]
train_y = training_dataset.iloc[:, 4]

# Setup feature columns
columns_feat = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]

# activation: sigmoid relu softmax
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(4, input_shape=(4,), activation='relu'),
  tf.keras.layers.Dense(10, activation='sigmoid'),
  tf.keras.layers.Dense(10, activation='sigmoid'),
  tf.keras.layers.Dense(3, activation='softmax'),
])

# loss: binary_crossentropy sparse_categorical_crossentropy 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200)

model.save('./models/petals2')