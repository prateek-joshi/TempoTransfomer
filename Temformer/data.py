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
        data = pd.read_csv(self.manifest_path)
        filenames = data['audio_filepath']
        self.train_filenames, self.val_filenames = train_test_split(filenames,test_size=self.split_ratio,shuffle=False)
        if self.shuffle:
            # Mention seed explicitly to preserve order
            self.train_filenames = tf.random.shuffle(self.train_filenames, seed=seed)
            self.val_filenames = tf.random.shuffle(self.val_filenames, seed=seed)
            
        # print(self.train_filenames[0], self.train_tempo[0],'\n',self.val_filenames[0],self.val_tempo[0])
        print(f'Found {len(self.train_filenames)} train samples\nFound {len(self.val_filenames)} validation samples')

    def get_waveform_and_label(self, filename):
        df = pd.read_csv(self.manifest_path)
        # print(df.iloc[0,:])
        filename = filename.numpy().decode().replace(r"[b']",'')
        tempo = df.loc[df['audio_filepath']==filename,'tempo'].to_numpy()[0]
        # print(filename, tempo)
        audio, _ = librosa.load(filename)
        # print('Converted')
        # print('1: ',audio.shape, type(audio), audio.dtype, type(tempo), tempo.dtype)
        return audio, tempo

    def get_waveform_generator(self):
        train_waveform_ds = tf.data.Dataset.from_tensor_slices(self.train_filenames).map(lambda filename: tf.py_function(self.get_waveform_and_label, [filename], [tf.float32,tf.float32]))
        val_waveform_ds = tf.data.Dataset.from_tensor_slices(self.val_filenames).map(lambda filename: tf.py_function(self.get_waveform_and_label, [filename], [tf.float32,tf.float32]))
        return train_waveform_ds, val_waveform_ds

    def get_spectrogram_and_label(self, waveform, tempo):
        # Cast the waveform tensors' dtype to float32.
        waveform = tf.cast(waveform, dtype=tf.float32)
        # Convert the waveform to a spectrogram via a STFT.
        # spectrogram = tf.signal.stft(
        #     waveform, frame_length=255, frame_step=128)
        spectrogram = librosa.cqt(waveform.numpy())
        # Obtain the magnitude of the CQT.
        spectrogram = tf.abs(spectrogram)
        # Add a `channels` dimension, so that the spectrogram can be used
        # as image-like input data with convolution layers (which expect
        # shape (`batch_size`, `height`, `width`, `channels`).
        spectrogram = spectrogram[..., tf.newaxis]
        return spectrogram, tempo

    def get_spectrogram_generator(self):
        self.get_data()
        wav_train, wav_val = self.get_waveform_generator()
        spec_train = wav_train.map(lambda waveform, tempo: tf.py_function(self.get_spectrogram_and_label, [waveform, tempo], [tf.float32, tf.float32]))
        spec_val = wav_val.map(lambda waveform, tempo: tf.py_function(self.get_spectrogram_and_label, [waveform, tempo], [tf.float32, tf.float32]))
        return spec_train, spec_val

    @staticmethod
    def plot_spectrogram(spectrogram, ax):
        if len(spectrogram.shape) > 2:
            assert len(spectrogram.shape) == 3
            spectrogram = np.squeeze(spectrogram, axis=-1)
        # Convert the frequencies to log scale and transpose, so that the time is
        # represented on the x-axis (columns).
        # Add an epsilon to avoid taking a log of zero.
        # log_spec = np.log(spectrogram.T + np.finfo(float).eps)
        log_spec = spectrogram
        height = log_spec.shape[0]
        width = log_spec.shape[1]
        X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

if __name__=='__main__':
    MANIFEST_PATH = 'E:\\TempoTransformer\\data\\final_manifest_new.csv'
    d = Dataset(MANIFEST_PATH, val_split=0.05)
    train,val = d.get_spectrogram_generator()

    rows = 3
    cols = 3
    n = rows*cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

    for i, (spectrogram, tempo) in enumerate(train.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        Dataset.plot_spectrogram(spectrogram.numpy(), ax)
        ax.set_title(tempo.numpy())
        ax.axis('off')

    plt.show()