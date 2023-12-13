import io
import torch
from fastapi import FastAPI, WebSocket
from text import text_to_sequence
from models import SynthesizerTrn
from text.symbols import symbols
from scipy.io.wavfile import write
import utils
import zipfile
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import os
from tqdm import tqdm
from data import CustomData
import concurrent.futures

ENGLISH_CONFIG = "./configs/ljs_base.json"
ENGLISH_MODEL = "./models/ljspeech.pth"
KIN_CONFIG = "./configs/rw_kin.json"
KIN_MODEL = "./models/"


app = FastAPI()


    

# Function to convert text to sequence
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    return torch.LongTensor(text_norm)

def generate_audio(text, hps, net_g):
    return get_audio(get_text(text, hps), net_g, hps)

def process_line(line, hps, net_g):
    return generate_audio(line, hps, net_g)


def get_audio(stn_tst,net_g,hps):
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


@app.websocket("/english/ws/text")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []
    hps = utils.get_hparams_from_file(ENGLISH_CONFIG)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(ENGLISH_MODEL, net_g, None)

    max_threads = os.cpu_count()


    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("FILE:"):
                # If the data starts with "FILE:", treat it as a file name
                file_data = data.replace("FILE:", "")
                file_lines = file_data.split("\n")


                
                # data = CustomData(file_lines, hps)
                # data_loader = torch.utils.data.DataLoader( data,  batch_size=5,num_workers=os.cpu_count())

                # for idx, (sid, stn_tst) in tqdm(data_loader):
                #     audio_data = await get_audio(stn_tst,net_g,hps)
                #     audio_files.append(audio_data)
                    
                # for line in file_lines:
                #     audio_data = get_audio(get_text(line, hps),net_g,hps)
                #     audio_files.append(audio_data)

                # audio_files.extend(process_lines(file_lines, hps, net_g, max_threads))

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                    # Process each line asynchronously using ThreadPoolExecutor
                    futures = {executor.submit(process_line, line, hps, net_g): line for line in file_lines}
                    
                    for future in concurrent.futures.as_completed(futures):
                        audio_data = future.result()
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
                text = data

                audio_data = get_audio(get_text(text, hps),net_g,hps)
                await websocket.send_bytes(audio_data)

                break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


@app.websocket("/rw/ws/text")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()
    audio_files = []
    
    hps = utils.get_hparams_from_file(KIN_CONFIG)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cuda()
    _ = net_g.eval()
    _ = utils.load_checkpoint(KIN_MODEL, net_g, None)

    max_threads = os.cpu_count()


    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("FILE:"):
                # If the data starts with "FILE:", treat it as a file name
                file_data = data.replace("FILE:", "")
                file_lines = file_data.split("\n")


                with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                    # Process each line asynchronously using ThreadPoolExecutor
                    futures = {executor.submit(process_line, line, hps, net_g): line for line in file_lines}
                    
                    for future in concurrent.futures.as_completed(futures):
                        audio_data = future.result()
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
                text = data

                audio_data = get_audio(get_text(text, hps),net_g,hps)
                await websocket.send_bytes(audio_data)

                break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
