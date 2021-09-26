import os
from io import StringIO

import PIL.Image
import streamlit as st
import librosa
import matplotlib.pyplot as plt
import numpy as np
from keras import models

from PIL import Image
import librosa.display

from analyze_model import gen_gif, predict

def plot_audio(filename):
    st.text(filename)
    data, sr = librosa.load(filename)
    fig, ax = plt.subplots()
    librosa.display.waveplot(data, sr=sr, ax=ax)
    ax.set_title(filename)
    st.pyplot(fig)

def plot_analyze(filename):
    my_gif = gen_gif(filename)
    image = Image.open(my_gif)
    prediction_text = predict(filename)
    st.image(image, prediction_text)

if __name__ == "__main__":
    st.sidebar.header("Входные характеристики")
    file = st.sidebar.file_uploader("Выбрать файл", type="wav", accept_multiple_files=False)
    if file is not None:
        with open(file.name, 'wb') as f:
            f.write(file.getvalue())
        plot_audio(file.name)
    btn = st.sidebar.button('Анализировать')

    if btn:
        if file is not None:
            plot_analyze(file.name)
            os.remove(file.name)


