
import torch
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
from utils import normalize_audio, HYPERPARAMS

def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform[0]  # Convert to 1D if needed

    noise = HYPERPARAMS["noise_factor"] * torch.randn_like(waveform)
    noisy_waveform = waveform + noise  # No need to normalize again

    chunk_size = HYPERPARAMS["chunk_size"]
    noisy_chunks = [noisy_waveform[i:i + chunk_size] for i in range(0, len(noisy_waveform) - chunk_size, chunk_size)]
    clean_chunks = [waveform[i:i + chunk_size] for i in range(0, len(waveform) - chunk_size, chunk_size)]

    noisy_tensor = torch.stack(noisy_chunks)
    clean_tensor = torch.stack(clean_chunks)

    dataset = TensorDataset(noisy_tensor, clean_tensor)
    dataloader = DataLoader(dataset, batch_size=HYPERPARAMS["batch_size"], shuffle=True)

    return dataloader, sample_rate, noisy_tensor, clean_tensor  # ✅ Now returning 4 values
