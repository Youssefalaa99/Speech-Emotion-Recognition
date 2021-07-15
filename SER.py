import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import os
import librosa
from playsound import playsound
from sklearn.model_selection import train_test_split

####################################################
# Global variables
emotions = {'ANG' : 0, 'DIS' : 1, 'FEA' : 2, 'HAP' : 3, 'NEU' : 4, 'SAD' : 5}
# data used for play_audio functions to get classes of the audio
data = {}
# zero-crossing rate and mel spectrogram feature spaces
x_zcr = []
x_mel = []
# labels of dataset
y = []  

####################################################
def scan_folder(parent):
  classes = []
  curr_word = "1001_DFA"
  # iterate over all the files in directory 'parent'
  for file_name in sorted(os.listdir(parent)):
    file_path = parent + '/' + file_name
    word = file_name[0:8]
    if curr_word != word:  
      data[curr_word] = classes
      classes = []
      curr_word = word
    classes.append(file_path)
    #zcr , mel = feature_creation(file_path)
    #x_zcr.append(zcr)
    #x_mel.append(mel)
    #y.append(emotions[file_path.split('/')[1][9:12]])

####################################################
def play_audio(audio):
  for x in data[audio]:
    print('Playing audio for file '+x.split('/')[1]+":\n")
    playsound(x)
    # read audio samples
    input_data = wavfile.read(x)
    audio = input_data[1]
    # plot the first 1024 samples
    plt.plot(audio)
    # label the axes
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    # set the title  
    plt.title(x.split('/')[1][0:12])
    # display the plot
    plt.show()
#play_audio("1001_DFA")

####################################################
def feature_creation(audio):
  x, sr = librosa.load(audio)
  zcr = librosa.feature.zero_crossing_rate(x)
  mel = librosa.feature.melspectrogram(x)
  #print(zcr.shape)
  #print(mel.shape)
  return zcr,mel

#feature_creation("D:\College\Term 8\Pattern Recognition\Assignment\Assignment 3\Dataset\Crema/1001_DFA_ANG_XX.wav")

####################################################
def train_model_zcr():
  #Splitting data 70% training , 30% testing
  x_zcr_train, x_zcr_test, y_zcr_train, y_zcr_test = train_test_split(x_zcr, y, test_size=0.3, random_state=42)
  #Splitting training data 5% validation
  x_zcr_train, x_zcr_test, valid_zcr_train, valid_zcr_test = train_test_split(x_zcr_train, y, test_size=0.05, random_state=42)

####################################################
def train_model_mel():
  #Splitting data 70% training , 30% testing
  x_mel_train, x_mel_test, y_mel_train, y_mel_test = train_test_split(x_mel, y, test_size=0.3, random_state=42)
  #Splitting training data 5% validation
  x_mel_train, x_mel_test, valid_mel_train, valid_mel_test = train_test_split(x_mel_train, y, test_size=0.05, random_state=42)

####################################################
#Main

if __name__ == '__main__':
  scan_folder('D:\College\Term 8\Pattern Recognition\Assignment\Assignment 3\Dataset\Crema')
  play_audio("1001_DFA")
  