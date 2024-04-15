import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import cv2
import glob
from sklearn.utils import shuffle
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from model import build_u2net

H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    train_x = sorted(glob.glob(os.path.join(path, 'train', 'blurred_image', '*jpg')))
    train_y = sorted(glob.glob(os.path.join(path, 'train', 'mask', '*.png')))
    #print(glob.glob(os.path.join(path, 'train\blurred_image', '*.jpg')))

    val_x = sorted(glob.glob(os.path.join(path, 'validation', 'P3M-500-NP', 'original_image', '*.jpg')))
    val_y = sorted(glob.glob(os.path.join(path, 'validation', 'P3M-500-NP', 'mask', '*.png')))
    return (train_x, train_y), (val_x, val_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  # [H, W, 3]
    x = cv2.resize(x, (W,H))
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  #[H,W]
    x = cv2.resize(x, (H,W))
    x = x/255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)  # [H, W, 1]
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch=2):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(tf_parse).batch(batch).prefetch(10)
    return ds
    


if __name__ == "__main__":
    
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir('files')

    # Hyperparameter
    batch_size = 8
    lr = 1e-4
    num_epochs = 10
    model_path = os.path.join('files', 'model.keras')
    csv_path = os.path.join('files', 'log.csv')

    # Dataset
    dataset_path = "C:/Users/priya/Documents/ImageMatting/P3M-10k"
    (train_x, train_y), (val_x, val_y) = load_dataset(dataset_path)
    print('Train : ', len(train_x), len(train_y))
    print('validation : ', len(val_x), len(val_y))

    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(val_x, val_y, batch=batch_size)

    # Model 
    model = build_u2net((H, W, 3))
    model.load_weights(model_path)
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr))

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
