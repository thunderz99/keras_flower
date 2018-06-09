import sys
import os
from image_model import ImageModel


model = ImageModel(batch_size=12, epochs=24, image_size=48, n=4, version=1)
model.prepare_train_data()
model.train()
# model.load("image_model.h5")

input_dir = "data/test"

for idx, category in enumerate(model.categories):
    category_dir = input_dir + "/" + category

    for file in os.listdir(category_dir):
        if file != ".DS_Store":
            filepath = category_dir + "/" + file
            result = model.predict(filepath)
            if result == category:
                print("[OK] " + result + ": ", filepath)
            else:
                print("[NG] " + result + ": ", filepath)


model.evaluate()
