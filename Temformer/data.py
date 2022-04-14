import os
import pathlib
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


class Dataset:
    def __init__(self, manifest_path, shuffle=True, val_split=0.1, batch_size=32) -> None:
        self.manifest_path = pathlib.Path(manifest_path)
        self.shuffle = shuffle
        self.split_ratio = val_split
        self.batch_size = batch_size

    def get_data(self):
        data = pd.read_pickle(self.manifest_path, compression=None)
        filenames = data['audio_filepath']
        self.train_filenames, self.val_filenames = train_test_split(filenames,test_size=self.split_ratio,shuffle=False)
        if self.shuffle:
            # Mention seed explicitly to preserve order
            self.train_filenames = tf.random.shuffle(self.train_filenames, seed=seed)
            self.val_filenames = tf.random.shuffle(self.val_filenames, seed=seed)
            
        # print(self.train_filenames[0], self.train_tempo[0],'\n',self.val_filenames[0],self.val_tempo[0])
        print(f'Found {len(self.train_filenames)} train samples\nFound {len(self.val_filenames)} validation samples')

    def get_waveform_and_label(self, filename):
        df = pd.read_pickle(self.manifest_path, compression=None)
        # print(df.iloc[0,:])
        filename = filename.numpy().decode().replace(r"[b']",'')
        tempo = df.loc[df['audio_filepath']==filename,'tempo'].to_numpy()[0]
        # print(filename, tempo)
        audio, _ = librosa.load(filename)
        # print('Converted')
        # print('1: ',audio.shape, type(audio), audio.dtype, type(tempo), tempo.dtype)
        return audio, tempo

    def get_waveform_generator(self):
        self.get_data()
        train_waveform_ds = tf.data.Dataset.from_tensor_slices(self.train_filenames).map(lambda filename: tf.py_function(self.get_waveform_and_label, [filename], [tf.float32,tf.float32]))
        val_waveform_ds = tf.data.Dataset.from_tensor_slices(self.val_filenames).map(lambda filename: tf.py_function(self.get_waveform_and_label, [filename], [tf.float32,tf.float32]))
        return train_waveform_ds, val_waveform_ds

if __name__=='__main__':
    MANIFEST_PATH = pathlib.Path('E:\\TempoTransformer\\data\\final_manifest.pkl')
    d = Dataset(MANIFEST_PATH, val_split=0.05)
    train,val = d.get_waveform_generator()

    rows = 3
    cols = 3
    n = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(train.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy()
        ax.set_title(label)

    plt.show()