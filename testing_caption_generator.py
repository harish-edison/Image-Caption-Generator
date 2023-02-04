import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

import string
import numpy as np
from PIL import Image
import os
from pickle import dump, load
import numpy as np

from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.utils import load_img
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.layers import concatenate
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

# small library for seeing the progress of loops.
from tqdm import tqdm_notebook as tqdm
tqdm().pandas()

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    return image

def extract_features(filename, model):
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
         if index == integer:
             return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text

filename='103106960_e8a41d64f8.jpg'
img_path = r'C:\Users\haris\Downloads\Image Caption Generator\Flickr8k_Dataset\Flicker8k_Dataset/103106960_e8a41d64f8.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model(r'C:\Users\haris\Downloads\Image Caption Generator\models/model_2.h5')
xception_model = Xception(include_top=False, pooling="avg")

photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")
print(description)
plt.imshow(img)