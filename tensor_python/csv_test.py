import tensorflow as tf
import pandas as pd

COLUMN_NAMES = [
        'SepalLength', 
        'SepalWidth',
        'PetalLength', 
        'PetalWidth', 
        'Species'
        ]

# Import testing dataset
test_dataset = pd.read_csv('./datasets/iris/iris_test.csv', names=COLUMN_NAMES, header=0, delimiter=";")
test_x = test_dataset.iloc[:, 0:4]
test_y = test_dataset.iloc[:, 4]

loaded_model = tf.keras.models.load_model('./models/petals')
loaded_model.evaluate(test_x,  test_y, verbose=2)