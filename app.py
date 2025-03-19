# from flask import Flask, request, send_file, render_template
# import os
# import torch
# import torchaudio
# from model import DenoisingAutoencoder
# from preprocess import load_audio
# from train import train_autoencoder
# from utils import HYPERPARAMS
# from io import BytesIO

# app = Flask(__name__)  

# # Path to save the uploaded and processed files
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'

# # Ensure the folders exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     # Check if the post request has the file part
#     if 'file' not in request.files:
#         return "No file part", 400
#     file = request.files['file']
#     if file.filename == '':
#         return "No selected file", 400
    
#     # Save the uploaded file
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)

#     # Process the audio file and generate denoised output
#     denoised_audio_path = process_audio(file_path)

#     # Send the denoised audio file as a response
#     return send_file(denoised_audio_path, as_attachment=True)

# def process_audio(input_file):
#     # Load training data
#     dataloader, sample_rate, noisy_tensor, clean_tensor = load_audio(input_file)

#     # Initialize and train the model
#     model = DenoisingAutoencoder()
#     model = train_autoencoder(model, dataloader, HYPERPARAMS["epochs"], HYPERPARAMS["learning_rate"])

#     # Generate Denoised Audio
#     denoised_audio_chunks = []
#     with torch.no_grad():
#         for noisy_chunk in noisy_tensor:
#             denoised_chunk = model(noisy_chunk.unsqueeze(0)).squeeze(0)
#             denoised_audio_chunks.append(denoised_chunk)

#     if len(denoised_audio_chunks) > 0:
#         denoised_audio = torch.cat(denoised_audio_chunks)
        
#         # Save the denoised audio
#         denoised_audio_path = os.path.join(PROCESSED_FOLDER, 'denoised_audio.wav')
#         torchaudio.save(denoised_audio_path, denoised_audio.unsqueeze(0), sample_rate)
        
#         return denoised_audio_path
#     else:
#         return "Error: Denoised audio is empty.", 400

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, send_file, render_template
import os
import torch
import torchaudio
from model import DenoisingAutoencoder
from preprocess import load_audio
from train import train_autoencoder
from utils import HYPERPARAMS
from io import BytesIO

app = Flask(__name__)

# Path to save the uploaded and processed files
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Process the audio file and generate denoised output
    denoised_audio_path = process_audio(file_path)

    # Send the denoised audio file as a response
    return send_file(denoised_audio_path, as_attachment=True)

def process_audio(input_file):
    # Load training data
    dataloader, sample_rate, noisy_tensor, clean_tensor = load_audio(input_file)

    # Initialize and train the model
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
        
        # Save the denoised audio
        denoised_audio_path = os.path.join(PROCESSED_FOLDER, 'denoised_audio.wav')
        torchaudio.save(denoised_audio_path, denoised_audio.unsqueeze(0), sample_rate)
        
        return denoised_audio_path
    else:
        return "Error: Denoised audio is empty.", 400

if __name__ == '__main__':
    app.run(debug=True)
