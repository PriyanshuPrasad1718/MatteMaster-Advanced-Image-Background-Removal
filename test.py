import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import glob
from tqdm import tqdm
import tensorflow as tf

W = 256
H = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)




if __name__ == '__main__':
    np.random.seed(42)
    tf.random.set_seed(42)


    for item in ['joint', 'mark']:
        create_dir(f"result/{item}")


    # Load model
    model_path = os.path.join('files', 'model.h5')
    model = tf.keras.models.load_model(model_path)

    # Dataset
    images = glob.glob('test/*')
    print(f"Images :{len(images)}")


    """   P   R   E   D   I   C   T   I   O   N   """

    for x in tqdm(images, total=len(images)):
        name = x.split('\\')[-1]
        print(name)

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = cv2.resize(image, (W,H))
        x = x/255.0
        #print(x.shape)
        x = np.expand_dims(x, axis=0)
        #print(x.shape)

        pred = model.predict(x, verbose=0)

        line = np.ones((H, 10, 3))*255

        pred_list = []
        for item in pred:
            p = item[0] * 255
            p = np.concatenate([p,p,p], axis=-1)
            pred_list.append(p)
            pred_list.append(line)

        save_image_path = os.path.join('result', 'mask', name)
        cat_image = np.concatenate(pred_list, axis=1)
        cv2.imwrite(save_image_path, cat_image)

        # Save final mask
        image_h, image_w, _ = image.shape

        y0 = pred[0][0]
        y0 = cv2.resize(y0, (image_w, image_h))
        y0 = np.expand_dims(y0, axis=-1)
        y0 = np.concatenate([y0, y0, y0], axis=-1)

        line = line = np.ones((image_h, 10, 3)) * 255

        cat_images = np.concatenate([image, line, y0*255, line, image*y0], axis=1)
        save_image_path = os.path.join("result", "joint", name)
        cv2.imwrite(save_image_path, cat_images)