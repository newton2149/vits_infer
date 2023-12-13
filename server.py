import os
import io

import torch
import commons
import utils
import random
import zipfile
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write




FRENCH_CONFIG = "configs/mlb_french.json"
ENGLISH_CONFIG = "configs/ljs_base.json"

FRENCH_MODEL = "G_69000.pth"
ENGLISH_MODEL = "models/ljspeech.pth"


def get_text(text, hps,lang):
    text_norm = text_to_sequence(text, hps.data.text_cleaners,lang)
    
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

## Inference
from fastapi import FastAPI

app = FastAPI()

def process_line(line,model,config):
    # Initialize model and parameters
    hps = utils.get_hparams_from_file(config)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model, net_g, None)

    # Convert text to sequence
    stn_tst = get_text(line, hps)

    # Generate audio with the model
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0].data.cpu().float().numpy()

    # Save audio to a file
    audio_filename = f"{random.randint(1, 1000000)}.mp3"
    write(audio_filename, hps.data.sampling_rate, audio)

    return audio_filename


@app.websocket("french/ws/zip")
async def zip_fr(websocket, path):

    # Handle incoming messages from clients
    while True:
        try:
            # Receive text file from the client
            file_data = await websocket.recv()

            # Create a temporary file
            with io.BytesIO() as temp_file:
                temp_file.write(file_data)
                temp_file.seek(0)

                # Read lines from the temporary file
                lines = temp_file.readlines()

                # Process each line and generate audio
                audio_filenames = []
                for line in lines:
                    audio_filename = process_line(line.decode("utf-8").strip(),FRENCH_MODEL,FRENCH_CONFIG)
                    audio_filenames.append(audio_filename)

                # Create a ZIP file containing the audio files
                with zipfile.ZipFile("audio.zip", "w") as zip_file:
                    for filename in audio_filenames:
                        zip_file.write(filename, arcname=filename)

                # Send the ZIP file back to the client
                with open("audio.zip", "rb") as zip_file:
                    zip_data = zip_file.read()
                await websocket.send(zip_data)

                # Remove temporary and audio files
                os.remove("audio.zip")
                for filename in audio_filenames:
                    os.remove(filename)

        except Exception as e:
            print(f"Error: {e}")
            break


@app.websocket("english/ws/zip")
async def zip_eng(websocket, path):

    # Handle incoming messages from clients
    while True:
        try:
            # Receive text file from the client
            file_data = await websocket.recv()

            # Create a temporary file
            with io.BytesIO() as temp_file:
                temp_file.write(file_data)
                temp_file.seek(0)

                # Read lines from the temporary file
                lines = temp_file.readlines()

                # Process each line and generate audio
                audio_filenames = []
                for line in lines:
                    audio_filename = process_line(line.decode("utf-8").strip(),ENGLISH_MODEL,ENGLISH_CONFIG)
                    audio_filenames.append(audio_filename)

                # Create a ZIP file containing the audio files
                with zipfile.ZipFile("audio.zip", "w") as zip_file:
                    for filename in audio_filenames:
                        zip_file.write(filename, arcname=filename)

                # Send the ZIP file back to the client
                with open("audio.zip", "rb") as zip_file:
                    zip_data = zip_file.read()
                await websocket.send(zip_data)

                # Remove temporary and audio files
                os.remove("audio.zip")
                for filename in audio_filenames:
                    os.remove(filename)

        except Exception as e:
            print(f"Error: {e}")
            break


@app.websocket("/french/ws/text")
async def fr_endpoint(websocket, path):

    # Initialize model and parameters
    hps = utils.get_hparams_from_file(FRENCH_CONFIG)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(FRENCH_MODEL, net_g, None)

    # Handle incoming messages from clients
    while True:
        try:
            # Receive text from the client
            text = await websocket.recv()

            # Convert text to sequence
            stn_tst = get_text(text, hps)

            # Generate audio with the model
            with torch.no_grad():
                x_tst = stn_tst.cuda().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
                audio = net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1,
                )[0][0, 0].data.cpu().float().numpy()

            # Send generated audio back to the client
            await websocket.send(audio.tobytes())

        except Exception as e:
            print(f"Error: {e}")
            break


@app.websocket("/english/ws/text")
async def eng_endpoint(websocket, path):

    # Initialize model and parameters
    hps = utils.get_hparams_from_file(ENGLISH_CONFIG)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cpu()
    _ = net_g.eval()
    _ = utils.load_checkpoint(ENGLISH_MODEL, net_g, None)

    while True:
        try:
            text = await websocket.recv()

            stn_tst = get_text(text, hps)

            with torch.no_grad():
                x_tst = stn_tst.cpu().unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
                audio = net_g.infer(
                    x_tst,
                    x_tst_lengths,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1,
                )[0][0, 0].data.cpu().float().numpy()

            await websocket.send(audio.tobytes())

        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
