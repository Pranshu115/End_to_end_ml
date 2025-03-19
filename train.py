import torch
import torch.optim as optim
from model import DenoisingAutoencoder, CombinedLoss

def train_autoencoder(model, dataloader, num_epochs, learning_rate):
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for noisy_audio, clean_audio in dataloader:
            optimizer.zero_grad()
            output = model(noisy_audio)
            loss = criterion(output, clean_audio)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")
    
    return model

if __name__ == "__main__":
    from preprocess import load_audio
    from utils import HYPERPARAMS

    # Load training data
    dataloader, _, _, _ = load_audio("Standard recording 16.mp3")  

    # Initialize and train model
    model = DenoisingAutoencoder()
    model = train_autoencoder(model, dataloader, HYPERPARAMS["epochs"], HYPERPARAMS["learning_rate"])

    # Save trained model
    torch.save(model.state_dict(), "denoising_autoencoder.pth")
    print("✅ Model saved as denoising_autoencoder.pth")
