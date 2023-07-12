import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from librosa.display import specshow

def create_spectrogram(y, sr, file_name, save_dir):
    # Create a spectrogram
    spectrogram = librosa.stft(y)

    # Convert amplitude spectrogram to dB-scaled spectrogram
    spect_db = librosa.amplitude_to_db(abs(spectrogram), ref=np.max)

    # Create a figure
    plt.figure(figsize=(12, 4))

    # Display the spectrogram
    specshow(spect_db, sr=sr, x_axis='time', y_axis='log')

    # Add a colorbar
    plt.colorbar(format='%+2.0f dB')

    # Set the title
    plt.title('Spectrogram of ' + file_name)

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'Spectrogram_' + file_name + '.png'))

def create_mfcc(y, sr, file_name, save_dir):
    # Create MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Create a figure
    plt.figure(figsize=(12, 4))

    # Display the MFCCs
    specshow(mfccs, sr=sr, x_axis='time')

    # Set the title
    plt.title('MFCC of ' + file_name)

    # Save the figure
    plt.savefig(os.path.join(save_dir, 'MFCC_' + file_name + '.png'))

# Directory where the audio files are located
folder_path = './dataset/other'

# Directory where to save the figures
save_dir = '/Users/junhoeum/Desktop/Summer_23/EQ_based_rec_algorithm/spectrum_images'

# Ensure the save_dir exists
os.makedirs(save_dir, exist_ok=True)

# Iterate over every audio file
for filename in os.listdir(folder_path):
    # Ensure that you process only audio files
    if filename.endswith('.wav') or filename.endswith('.mp3'):
        # Load the audio file
        y, sr = librosa.load(os.path.join(folder_path, filename))

        # Call the function to create and save the spectrogram
        create_spectrogram(y, sr, filename, save_dir)

        # Call the function to create and save the MFCC
        create_mfcc(y, sr, filename, save_dir)
