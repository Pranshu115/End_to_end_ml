import torch

def normalize_audio(audio):
    return audio / torch.max(torch.abs(audio))

HYPERPARAMS = {
    "sample_rate": 16000,
    "n_fft": 2048,
    "hop_length": 512,
    "learning_rate": 0.001,
    "batch_size": 2,
    "epochs": 100,
    "noise_factor": 0.3,
    "chunk_size": 16384
}
