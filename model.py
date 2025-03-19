
import torch
import torch.nn as nn
import torchaudio.transforms as T

def normalize_audio(audio):
    return audio / torch.max(torch.abs(audio))
def normalize_audio(audio):
    return (audio - audio.mean()) / (audio.std() + 1e-6)  # Standardization

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=2, padding=4),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.ConvTranspose1d(32, 1, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1)

class CombinedLoss(nn.Module):
    def __init__(self, n_fft=512):
        super(CombinedLoss, self).__init__()
        self.stft = T.Spectrogram(n_fft=n_fft, power=2)
        self.mse = nn.MSELoss()
    
    def forward(self, output, target):
        output_spec = self.stft(output)
        target_spec = self.stft(target)
        return self.mse(output, target) + self.mse(output_spec, target_spec)
