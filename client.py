import io
import torch
from fastapi import FastAPI, WebSocket
from text import text_to_sequence
from models import SynthesizerTrn
from text.symbols import symbols
from scipy.io.wavfile import write
import utils
import commons
import random
import zipfile
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write

ENGLISH_CONFIG = "./configs/ljs_base.json"
ENGLISH_MODEL = "./models/ljspeech.pth"
KIN_CONFIG = "./configs/rw_kin.json"
KIN_MODEL = "./models/"


app = FastAPI()

# Function to convert text to sequence
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    return torch.LongTensor(text_norm)

# Function to process text and generate audio
async def process_text(text, model, config):
    hps = utils.get_hparams_from_file(config)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(model, net_g, None)

    stn_tst = get_text(text, hps)
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

    # Save audio to a file (temporary here, you can modify as needed)
    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

# WebSocket endpoint for English text-to-audio
@app.websocket("/english/ws/text")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()
    audio_files = []
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("FILE:"):
                # If the data starts with "FILE:", treat it as a file name
                file_data = data.replace("FILE:", "")
                file_lines = file_data.split("\n")
                for line in file_lines:
                    audio_data = await process_text(line, ENGLISH_MODEL, ENGLISH_CONFIG)
                    audio_files.append(audio_data)

                with zipfile.ZipFile("audio.zip", "w") as zip_file:
                    for idx, audio_data in enumerate(audio_files):
                        zip_file.writestr(f"audio_{idx}.wav", audio_data)

                with open("audio.zip", "rb") as zip_file:
                    zip_data = zip_file.read()
                await websocket.send_bytes(zip_data)

                #delete zip file
                os.remove("audio.zip")

                break
                        
            else:
                text = data.strip()

                audio_data = await process_text(text, ENGLISH_MODEL, ENGLISH_CONFIG)
                await websocket.send_bytes(audio_data)

                break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


@app.websocket("/rw/ws/text")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()
    audio_files = []
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("FILE:"):
                # If the data starts with "FILE:", treat it as a file name
                file_data = data.replace("FILE:", "")
                file_lines = file_data.split("\n")
                for line in file_lines:
                    audio_data = await process_text(line, KIN_MODEL, KIN_CONFIG)
                    audio_files.append(audio_data)

                with zipfile.ZipFile("audio.zip", "w") as zip_file:
                    for idx, audio_data in enumerate(audio_files):
                        zip_file.writestr(f"audio_{idx}.wav", audio_data)

                with open("audio.zip", "rb") as zip_file:
                    zip_data = zip_file.read()
                await websocket.send_bytes(zip_data)

                #delete zip file
                os.remove("audio.zip")

                break
                        
            else:
                text = data.strip()

                audio_data = await process_text(text, KIN_MODEL, KIN_CONFIG)
                await websocket.send_bytes(audio_data)

                break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
