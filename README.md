## Audio texture synthesis (and a bit of stylization)

This is an extension of [texture synthesis](https://arxiv.org/abs/1505.07376) and [style transfer](https://arxiv.org/abs/1508.06576) method of Leon Gatys et al. based on [Justin Johnson's code](https://github.com/jcjohnson/neural-style) for neural style transfer.

To listen to examples go to the blog post. Almost identical Lasagne implementation by [Vadim Lebedev](http://sites.skoltech.ru/compvision/members/vadim-lebedev/) can be found [here](https://github.com/vadim-v-lebedev/audio_style_tranfer). Also check out [TensorFlow implementation](https://github.com/DmitryUlyanov/neural-style-audio-tf).

### Examples
As there is no way to embed an audio player with github markdown please follow [this link]() for the examples of texture synthesis and style transfer.

### Prerequisites
- [Torch7](http://torch.ch/docs/getting-started.html#_)
  - With [npy4th](https://github.com/htwaijry/npy4th), [cudnn.torch](https://github.com/soumith/cudnn.torch)
- Python (tested with 2.7)
  - numpy, [librosa](https://github.com/librosa/librosa), [torchfile](https://github.com/bshillingford/python-torchfile)
- Download pretrained Network and mean file

```
wget https://www.dropbox.com/s/xpyoehayuhxvibq/net.t7?dl=1 -O data/net.t7
wget https://www.dropbox.com/s/dwsq33r5bsgy9cd/mean.t7?dl=1 -O data/mean.t7
```

### Usage
##### 1. First convert raw audio to spectrogram

```
python get_spectrogram.py --in_audio <path to audio file> --out_npy <where to save spectrogram>
```

Additional arguments:
- `-offset` and `-duration` control boundaries of a segment which will be converted to spectrogram. [By default first 10 seconds are used].
- `-sr`: sample rate (samples), [default = 44100].
- `-n_fft`: FFT window size (samples) [default = 1024].

##### 2. Run texture synthesis or style transfer

This is the cmd used for all the texture synthesis examples:
```
th neural_style_audio.lua -content <path to content.npy> -style <path to style.npy> -content_layers <indices of content layers> -style_layers <indices of style layers> -style_weight <number> -content_weight <number>
```
Parameters:
- `-content`, `-style` are paths to `.npy` spectrogram files generated previously.
- `-content_layers`, `-style_layers`: comma separated layer indices (the provided net has about 50 layers and of encoder-decoder type).
- `-style_weight`, `-content_weight`: set `content_weight` to zero for texture synthesis.
- `-lowres`: the higher receptive field the larger textures network captures. The easiest way to increase a receptive field is to drop every other column from an input spectrogram (kind of downscaling). Use this flag to process spectrogram both in original and low resolution at the same time.
- `-how_div`, `-normalize_gradients`, `-loss` change some details of loss calculation.
- `-save` indicates folder to save intermediate results.

##### 3. Convert spectrogram back to audio file
```
python invert_spectrogram.py --spectrogram_t7 <path to synthesized spectrogram> --out_audio keyboard2_texture.wav
```
Parameters:

### Command-line used for texture examples

```
python get_spectrogram.py --out_npy data/inputs/keyboard2.npy --in_audio data/inputs/keyboard2.mp3
th neural_style_audio.lua -style data/inputs/keyboard2.npy -content_layers '' -style_layers 1,5,10,15,20,25 -style_weight 10000000 -optimizer lbfgs -learning_rate 1e-1 -num_iterations 5000 -lowres
python invert_spectrogram.py --spectrogram_t7 data/out/out.png.t7 --out_audio data/out/keyboard2_texture.wav
```

### Command-line used for style transfer

We have also implemented minimalistic script identical to TensorFlow and Lasagne scripts. Here is example how to use it:

```
python get_spectrogram.py --out_npy data/inputs/usa.npy --in_audio data/inputs/usa.mp3 --n_fft 2048 --sr 22050
python get_spectrogram.py --out_npy data/inputs/imperial.npy --in_audio data/inputs/imperial.mp3 --n_fft 2048 --sr 22050
th neural_style_audio_random.lua -alpha 1e-2
python invert_spectrogram.py --spectrogram_t7 out.t7 --out_audio out.wav --n_iter 500 --n_fft 2048 --sr 22050
```
