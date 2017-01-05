import numpy as np
import librosa
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--out_npy', help='Where to store the spectrogram.')
parser.add_argument('--in_audio', help='Path to raw audio (wav, mp3, etc.).')
parser.add_argument('--offset', type=float, default=0,
                    help='Start from (sec).')
parser.add_argument('--duration', type=float, default=10,
                    help='Desired duration (sec).')
parser.add_argument('--sr', type=int, default=44100,
                    help='Sample rate to use.')
parser.add_argument('--n_fft', type=int, default=1024,
                    help='FFT Window length.')

args = parser.parse_args()


# Read audio
y, _ = librosa.core.load(args.in_audio, sr=args.sr,
                         offset=args.offset, duration=args.duration)

# STFT
D = librosa.stft(y, n_fft=args.n_fft)
S, phase = librosa.magphase(D)

# S = log(S+1)
S = np.log1p(S)

np.save(args.out_npy, S)
