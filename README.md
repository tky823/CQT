# CQT
Constant-Q transform for PyTorch.

## Requirements

```sh
pip install -r requirements.txt
```

NOTE: You may need to install FFmpeg to use `torchaudio.load`.

## Example

```python
>>> import torch
>>> import torchaudio
>>> import torchaudio.transforms as aT
>>> from utils import note_to_hz
>>> from cqt import ConstantQTransform
>>> path = "https://pytorch-tutorial-assets.s3.amazonaws.com/VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
>>> waveform, sample_rate = torchaudio.load(path)
>>> waveform.size()
torch.Size([1, 54400])
>>> sample_rate
16000
>>> f_nyq = sample_rate / 2
>>> f_nyq
8000.0
>>> f_min, hop_length, n_bins, bins_per_octave = note_to_hz("C1"), 512, 48, 12  # parameters of CQT
>>> f_min * (2 ** (n_bins / bins_per_octave))  # f_max
523.2511306011972  # f_max, which is lower than f_nyq by a large margin.
>>> transform = ConstantQTransform(sample_rate, hop_length=hop_length, f_min=f_min, n_bins=n_bins, bins_per_octave=bins_per_octave)
>>> spectrogram = transform(waveform)
>>> spectrogram.size()
torch.Size([1, 48, 107])
>>> # for computational efficiency
>>> new_sample_rate = 4000  # f_max is less than new_sample_rate / 2
>>> new_hop_length = hop_length // (sample_rate // new_sample_rate)
>>> new_hop_length
128
>>> resampler = aT.Resample(sample_rate, new_sample_rate)
>>> new_waveform = resampler(waveform)
>>> new_waveform.size()
torch.Size([1, 13600])
>>> new_transform = ConstantQTransform(new_sample_rate, hop_length=new_hop_length, f_min=f_min, n_bins=n_bins, bins_per_octave=bins_per_octave)
>>> new_spectrogram = new_transform(new_waveform)
>>> new_spectrogram.size()
torch.Size([1, 48, 107])
>>> torch.mean(torch.abs(new_spectrogram - spectrogram))
tensor(0.0002)
```
