import os
import torch
import torchaudio
from model import DenoisingAutoencoder
from preprocess import load_audio
from train import train_autoencoder
from utils import HYPERPARAMS

def process_audio(input_file):
    dataloader, sample_rate, noisy_tensor, clean_tensor = load_audio(input_file)

    print(f"✅ Loaded audio: Sample rate = {sample_rate}, Noisy tensor shape = {noisy_tensor.shape}")

    # Save Noisy Audio (Check if it is valid)
    noisy_audio_path = os.path.abspath("noisy_audio.wav")
    if noisy_tensor.shape[0] > 0:  # Ensure it's not empty
        torchaudio.save(noisy_audio_path, noisy_tensor.squeeze(1), sample_rate)
        print(f"✅ Noisy audio saved at {noisy_audio_path}")
    else:
        print("❌ Noisy audio tensor is empty. Check the preprocessing step.")

    # Train Model
    model = DenoisingAutoencoder()
    model = train_autoencoder(model, dataloader, HYPERPARAMS["epochs"], HYPERPARAMS["learning_rate"])
    
    # Generate Denoised Audio
    denoised_audio_chunks = []
    with torch.no_grad():
        for noisy_chunk in noisy_tensor:
            denoised_chunk = model(noisy_chunk.unsqueeze(0)).squeeze(0)
            denoised_audio_chunks.append(denoised_chunk)

    if len(denoised_audio_chunks) > 0:
        denoised_audio = torch.cat(denoised_audio_chunks)
        print("✅ Denoised audio generated successfully!")

        # Save Denoised Audio
        denoised_audio_path = os.path.abspath("denoised_audio.wav")
        torchaudio.save(denoised_audio_path, denoised_audio.unsqueeze(0), sample_rate)
        print(f"✅ Denoised audio saved at {denoised_audio_path}")
    else:
        print("❌ Denoised audio is empty. Model might not be working correctly.")

if __name__ == "__main__":
    process_audio("Standard recording 16.mp3")
