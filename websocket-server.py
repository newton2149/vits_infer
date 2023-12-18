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
import commons
import zipfile
from torch.utils.data import  DataLoader

ENGLISH_CONFIG = "./configs/ljs_base.json"
ENGLISH_MODEL = "./models/ljspeech.pth"
KIN_CONFIG = "./configs/rw_kin.json"
KIN_MODEL = "./models/"


app = FastAPI()
    


def extract_and_read_text(zip_file_path, text_file_name):

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:

        zip_ref.extract(text_file_name)  


    extracted_text = []
    with open(text_file_name, 'r') as file:
        extracted_text = file.readlines()  
    
    return extracted_text
# Function to convert text to sequence
def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    return torch.LongTensor(text_norm)


def generate_audio(text, hps, net_g):
    return get_audio(get_text(text, hps), net_g, hps)


def generate_audio_cpu(text, hps, net_g):
    return get_audio_cpu(get_text(text, hps), net_g, hps)

def process_line(line, hps, net_g):
    return generate_audio(line, hps, net_g)

def process_line_cpu(line, hps, net_g):
    return generate_audio_cpu(line, hps, net_g)


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

def get_audio_cpu(stn_tst,net_g,hps):
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

    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()


@app.websocket("V1/english/ws/text/gpu")
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


                with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                    # Process each line asynchronously using ThreadPoolExecutor
                    futures = {executor.submit(process_line, line, hps, net_g): line for line in file_lines}
                    
                    for future in concurrent.futures.as_completed(futures):
                        audio_data = future.result()
                        audio_files.append(audio_data)

                with zipfile.ZipFile("audio_gpu.zip", "w") as zip_file:
                    for idx, audio_data in enumerate(audio_files):
                        zip_file.writestr(f"audio_{idx}.wav", audio_data)

                with open("audio_gpu.zip", "rb") as zip_file:
                    zip_data = zip_file.read()
                await websocket.send_bytes(zip_data)

                #delete zip file
                os.remove("audio_gpu.zip")

                break
                        
            else:
                text = data

                audio_data = get_audio(get_text(text, hps),net_g,hps)
                await websocket.send_bytes(audio_data)

                break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


@app.websocket("/english/ws/text/gpu")
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
                print(len(file_lines))

                data = CustomData(file_lines, hps)
                

                data_loader = data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

                for id,stn_tst, in tqdm(data_loader):
                    print(id)
                    for stn_tst in stn_tst:
                        audio_data = get_audio(stn_tst,net_g,hps)
                        audio_files.append(audio_data)


                

                with zipfile.ZipFile("audio_gpu.zip", "w") as zip_file:
                    for idx, audio_data in enumerate(audio_files):
                        zip_file.writestr(f"audio_{idx}.wav", audio_data)

                with open("audio_gpu.zip", "rb") as zip_file:
                    zip_data = zip_file.read()
                await websocket.send_bytes(zip_data)

                #delete zip file
                os.remove("audio_gpu.zip")

                
                        
            else:
                text = data

                audio_data = get_audio(get_text(text, hps),net_g,hps)
                await websocket.send_bytes(audio_data)

                break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()





@app.websocket("/english/ws/text/cpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []
    hps = utils.get_hparams_from_file(ENGLISH_CONFIG)
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model
    ).cpu()
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


                with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                    # Process each line asynchronously using ThreadPoolExecutor
                    futures = {executor.submit(process_line_cpu, line, hps, net_g): line for line in file_lines}
                    
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
