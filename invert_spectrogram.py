import numpy as np
import librosa
import torchfile
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--spectrogram_t7', help='Path to t7 spectrogram.')
parser.add_argument('--out_audio', help='Where to save audio file.')
parser.add_argument('--sr', type=int, default=44100,
                    help='Sample rate.')
parser.add_argument('--n_fft', type=int, default=1024,
                    help='FFT window length.')
parser.add_argument('--n_iter', type=int, default=2000,
                    help='Number of iterations to make.')

args = parser.parse_args()

spectr = torchfile.load(args.spectrogram_t7)
S = np.zeros([args.n_fft / 2 + 1, spectr.shape[1]])
S[:spectr.shape[0]] = spectr


def update_progress(progress):
    print "\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(progress * 50),
                                                  progress * 100),


def phase_restore(mag, random_phases, N=50):
    p = np.exp(1j * (random_phases))

    for i in range(N):
        _, p = librosa.magphase(librosa.stft(
            librosa.istft(mag * p), n_fft=args.n_fft))
        update_progress(float(i) / N)
    return p

random_phase = S.copy()
np.random.shuffle(random_phase)
p = phase_restore((np.exp(S) - 1), random_phase, N=args.n_iter)

# ISTFT
y = librosa.istft((np.exp(S) - 1) * p)

librosa.output.write_wav(args.out_audio, y, args.sr, norm=False)
