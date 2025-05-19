import numpy as np
import torch
from torchaudio.transforms import Resample, MelSpectrogram

import src.params as params

class WaveformToMelSpec(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        
        window_length_in_samples = int(round(params.SAMPLE_RATE * params.STFT_WINDOW_LENGTH))
        hop_length_in_samples = int(round(params.SAMPLE_RATE * params.STFT_HOP_LENGTH))
        fft_length = 2 ** int(np.ceil(np.log(window_length_in_samples) / np.log(2.0)))
        
        self.transform_to_mel = LogMelSpecTrans(
            params.SAMPLE_RATE,
            n_fft=fft_length,
            win_length=window_length_in_samples,
            hop_length=hop_length_in_samples,
            f_min=params.MEL_MIN_HZ,
            f_max=params.MEL_MAX_HZ,
            n_mels=params.NUM_MEL_BANDS
        )
        
        self.device = device
        if self.device is not None:
            self.to(self.device)
    
    def __call__(self, waveform, sample_rate):
        x = waveform.mean(axis=0, keepdims=True)
        
        resampler = Resample(sample_rate, params.SAMPLE_RATE)
        resampler.to(self.device)
        
        # Resample and generate mel spectrogram
        x = resampler(x)
        x = self.transform_to_mel(x)
        x = x.squeeze(dim=0).T # (1, C, T) --> (T, C)
        mel_spectrogram = x.cpu().numpy().copy() # for saving spectrogram as images
        
        # Split into chunks
        window_size = int(round(params.PATCH_WINDOW_LENGTH / params.STFT_HOP_LENGTH))
        if params.PATCH_HOP_LENGTH == params.PATCH_WINDOW_LENGTH: # no overlap
            num_chunks = x.shape[0] // window_size
            num_frames = num_chunks * window_size
            x = x[:num_frames]
            x = x.reshape(num_chunks, 1, window_size, x.shape[-1])
        else:
            patch_hops = int(round(params.PATCH_HOP_LENGTH / params.STFT_HOP_LENGTH))
            num_chunks = max((x.shape[0] - window_size) // patch_hops + 1, 1)
            num_frames = window_size + (num_chunks - 1) * patch_hops
            x = x[:num_frames]
            x_in_frames = x.reshape(-1, x.shape[-1])
            
            pad_size = max(0, window_size - (x_in_frames.shape[0] % window_size))
            if pad_size > 0:
                x_in_frames = torch.nn.functional.pad(x_in_frames, (0, 0, 0, pad_size))
            num_chunks = (x_in_frames.shape[0] - window_size) // patch_hops + 1
            
            x_out = np.empty((num_chunks, window_size, x.shape[-1]))
            for i in range(num_chunks):
                start_frame = i * patch_hops
                x_out[i] = x_in_frames[start_frame : start_frame + window_size].cpu()
            x = x_out.reshape(num_chunks, 1, window_size, x.shape[-1])
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            
        return x, mel_spectrogram
        
        
class LogMelSpecTrans(MelSpectrogram):
    def forward(self, waveform, device=None):
        if device is not None:
            self.to(device)
            
        spectrogram = self.spectrogram(waveform)
        spectrogram = spectrogram ** 0.5
        
        mel_spectrogram = self.mel_scale(spectrogram)
        mel_spectrogram = torch.log(mel_spectrogram + params.LOG_DELTA)
        
        return mel_spectrogram
        