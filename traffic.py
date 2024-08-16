import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 100
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):

    print("Loading images and labels...")

    images, labels = [], []

    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, data_dir)
    print(f"Main Folder acessed now is the folder: {data_dir}")
    # Abrindo a pasta data_dir 
    if os.path.exists(data_dir):
        for folder in os.listdir(data_dir):
            print(f"Acessing Subfolder: {folder}")
            current_sub_dir = os.path.join(data_dir, folder)
            if folder != ".DS_Store":                
                for file in os.listdir(current_sub_dir):
                    labels.append(int(folder))
                    image = cv2.imread(os.path.join(current_sub_dir, file))
                    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                    image = image_resized / 255.0
                    images.append(image_resized)
    else:          
        print(f"Invalid directory")

    print("Images and labels read successfully")
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")
    print(f"Unique Labels: {len(set(labels))}")
    return (images, labels)

def get_model():

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),

        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
