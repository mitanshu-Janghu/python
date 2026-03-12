import time
import numpy as np

# change according to your model type
from tensorflow.keras.models import load_model

# load model
model = load_model("model.h5")

print("Model loaded successfully")

while True:
    try:
        # simulate input data (replace with real input)
        input_data = np.random.rand(1, 224, 224, 3)

        # run prediction
        prediction = model.predict(input_data)

        print("Prediction:", prediction)

        # wait before next run
        time.sleep(1)

    except Exception as e:
        print("Error:", e)