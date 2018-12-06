import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import math
import random

def pitch_shift(y, sr):
    sr_start = sr * 30
    print(sr_start)
    print(sr_start + 3*sr)
    y_pitch_shift = librosa.effects.pitch_shift(y[sr_start:(sr_start + sr*5)], sr, n_steps=6)

    y[sr_start:(sr_start + sr*5)] = y_pitch_shift

    return librosa.output.write_wav("lol.wav",y, sr)

def bass_operation(y, sr):
    D = librosa.stft(y)
    bin_size =  sr / 2048
    start_bin = math.floor(60 / bin_size)
    end_bin = math.floor(260 / bin_size)
    for bin in range(start_bin, end_bin):
        if bin in range(start_bin + 2):
            D[bin,:] *= 8
        if bin in range(start_bin + 2, start_bin + 5):
            D[bin, :] *= 12
        if bin in range(start_bin + 5, end_bin):
            D[bin, :] *= 8
    
    y_bass = librosa.istft(D)

    return librosa.output.write_wav("lol.wav",y_bass, sr)
	
def overlay(y, sr):
    beats = librosa.beat.beat_track(y=y, sr=sr, units="time")[1]
    echo_volume = [0.8, 0.7, 0.6]
    idx = random.randint(30,35)
    #print(beats[idx])
    start_idx =  int((int(beats[idx]) * sr))
    end_idx = int((int(beats[idx+20]* sr)))
    #print(start_idx - end_idx)
    
    y_new = []
    for i in range (int(start_idx), int(end_idx)):
        y_new.append(y[i])
    y_effect = []
    for vol in echo_volume:
        y_echo_vol = np.multiply(y_new, vol)
        y_effect = np.append(y_effect, y_echo_vol)
        
    print(y_effect.shape)
    y_echo = np.pad(y_effect, (start_idx+(end_idx-start_idx), len(y)-(end_idx+len(y_effect))), "constant")
    y_new = y + y_echo
    
    return librosa.output.write_wav("lol.wav",y_new, sr)
    
