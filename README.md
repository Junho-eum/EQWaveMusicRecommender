## EQ_based_rec_algorithm
Music recommendation algorithm based on EQ soundwave
In the current age of streaming services, a personalized and efficient music recommendation system holds great importance. Existing methods typically rely on collaborative filtering, content-based features, and hybrid systems. This study introduces an approach using EQ sound wave pattern analysis to enhance the accuracy of music recommendations, aiming to provide a richer and more personalized user experience.
 
## Background and Related Work

Previous studies have used a variety of methods for music recommendation, ranging from collaborative filtering, which bases recommendations on the listening habits of similar users, to content-based filtering, which recommends songs with similar features. This research aims to expand upon these methodologies by incorporating EQ sound wave analysis, a largely unexplored aspect in music recommendation systems.

## Methodology

- The proposed algorithm utilizes a dataset consisting of 20 songs from various genres, each represented by a 15-second EQ wave pattern recording of the intro section. We applied frequency-domain analysis to these recordings to develop a unique EQ sound wave 'fingerprint' for each song. The user's listening frequency for each song was used to train a machine learning model to predict their preference for similar EQ wave patterns.
![user_input_1](https://github.com/Junho-eum/EQ_based_rec_algorithm/assets/74083204/ecb4b972-23f7-49a6-9093-169252bc9df1)
![user_input_1](https://github.com/Junho-eum/EQ_based_rec_algorithm/assets/74083204/21e74b06-7df9-49bc-b03e-72f497fe3a88)

## Requirements

  ```
  Python 3.6 or above
  PyQt5
  python_speech_features
  seaborn
  librosa
  numpy
  fastdtw
  glob
  matplotlib
  ```

- An audio dataset in the .mp3 format divided into various genres or types, placed in the ./dataset/ directory

## Instructions
  1. Install the required Python libraries by running pip install PyQt5 python_speech_features seaborn librosa numpy fastdtw matplotlib.
  
  2. Clone this repository to your local machine.
  
  3. Place your audio dataset in the ./dataset/ directory. The dataset should be organized into sub-directories for each genre or type of song.
  
  4. Run the Python script run recommendation_algo_multi_input.py.
  
  5. The application will display a list of all songs available in your dataset.
  
  6. Double-click on a song from the list to select it. This will be the song that the system will use as a reference for recommendations.
  
  7. After you have selected a songs you like, click on the 'Exit' button. The system will then begin processing and comparing the selected song against the rest of the dataset.
  
  Wait for the processing to complete. The system will then display the song that is most similar to your selected song based on the MFCC features.

## How it works
- This system uses MFCCs to convert the audio tracks into a form that can be processed. The FastDTW algorithm is then used to compare the MFCCs of the selected song against the MFCCs of the other songs in the dataset. The song with the smallest distance measure, which indicates the highest similarity to the selected song, is then recommended.

- This application uses PyQt5 to provide a simple GUI for song selection and result display. It uses the librosa library to handle audio file loading and processing.

## Note
- The system currently supports .mp3 format for audio files. Ensure your dataset consists of .mp3 files.

- This code uses a fixed length for MFCC data padding or truncation. Ensure that the audio tracks in your dataset are compatible with the chosen fixed length. The current fixed length is set to 661500.
## DTW Algorithm appication (Dynamic Time Warping)

## Library for K pop recommendation using 15 DTW
