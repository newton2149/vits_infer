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


hps = utils.get_hparams_from_file(ENGLISH_CONFIG)
net_g_gpu = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).cuda()
_ = net_g_gpu.eval()
_ = utils.load_checkpoint(ENGLISH_MODEL, net_g_gpu, None)



net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model
).cpu()
_ = net_g.eval()
_ = utils.load_checkpoint(ENGLISH_MODEL, net_g, None)

max_threads = os.cpu_count()


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    return torch.LongTensor(text_norm)


def generate_audio(text, hps, net_g):
    return get_audio(get_text(text, hps), net_g, hps)


def generate_audio_cpu(text, hps, net_g):
    return get_audio_cpu(get_text(text, hps), net_g, hps)




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


@app.websocket("/english/gpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()



    try:
        while True:
            data = await websocket.receive_text()
            
                        

            text = data

            audio_data = get_audio(get_text(text, hps),net_g_gpu,hps)
            await websocket.send_bytes(audio_data)

            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()





@app.websocket("/english/cpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()            
                        

            text = data

            audio_data = get_audio_cpu(get_text(text, hps),net_g,hps)
            await websocket.send_bytes(audio_data)

            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
