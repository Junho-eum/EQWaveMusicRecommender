from python_speech_features import mfcc
import seaborn as sns
import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
import glob

def pad_or_truncate(data, fixed_length):
    if data.shape[1] > fixed_length:
        return data[:,:fixed_length]
    elif data.shape[1] < fixed_length:
        padding = np.zeros((data.shape[0], fixed_length - data.shape[1]))
        return np.concatenate((data, padding), axis=1)
    return data

dirs = ['./dataset/Classical_15_sec', './dataset/county 15 sec', './dataset/other', './dataset/trap']
songs_dict = {}

for dir in dirs:
    print(f"\nSongs in {dir.split('/')[-1]}:")
    tracks = glob.glob(dir + "/*.mp3")
    for track in tracks:
        song_name = track.split('/')[-1]
        print(song_name)
        songs_dict[song_name] = track

user_songs = []

while True:
    song_choice = input("\nEnter the name of the song you like or type 'exit' to finish: ")
    if song_choice.lower() == 'exit':
        break
    else:
        user_songs.append(songs_dict[song_choice])

similarity_values = []
fixed_length = 661500

for user_song in user_songs:
    user_data, user_sample_rate = librosa.load(user_song, sr=44100)
    user_a = user_data.T[500:-500]
    user_a = user_a / user_a.max()
    user_mfcc_feat = mfcc(user_a, user_sample_rate, nfft=2048)
    user_mfcc_data = user_mfcc_feat.T
    user_mfcc_data = pad_or_truncate(user_mfcc_data, fixed_length)
    for dir in dirs:
        tracks = glob.glob(dir + "/*.mp3")
        for track in tracks:
            if track not in user_songs:  # Only compare against non-selected songs
                data, sample_rate = librosa.load(track, sr=44100)
                a = data.T[500:-500]
                a = a / a.max()
                mfcc_feat = mfcc(a, sample_rate, nfft=2048)
                mfcc_data = mfcc_feat.T
                mfcc_data = pad_or_truncate(mfcc_data, fixed_length)
                dis, _ = fastdtw(user_mfcc_data, mfcc_data)
                similarity_values.append((dis, track))

similarity_values.sort(key=lambda x: x[0])

print("\nThe most similar song to your selection is: ", similarity_val ues[0][1])
