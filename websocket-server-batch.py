import io
import torch
from fastapi import FastAPI, WebSocket
from text import text_to_sequence
from models import SynthesizerTrn
from text.symbols import symbols
from scipy.io.wavfile import write
import utils
import zipfile
import commons
from models import SynthesizerTrn
from text.symbols import symbols
from text.symbols import symbols
from text import text_to_sequence

from text.mlb_fr_symbols import symbols as fr_symbols
from text.mlb_fr import text_to_sequence as fr_text_to_sequence

from text.vctk_symbols import symbols as vctk_symbols
from text.vctk import text_to_sequence as vctk_text_to_sequence

from text.rw_symbols import symbols as rw_symbols
from text.rw import text_to_sequence as rw_text_to_sequence
from scipy.io.wavfile import write
import os
from tqdm import tqdm
from data import CustomData
import zipfile
from torch.utils.data import  DataLoader



GPU = torch.cuda.is_available()

ENGLISH_CONFIG = "./configs/ljs_base.json"
ENGLISH_MODEL = "./models/ljspeech.pth"

KIN_CONFIG = "./configs/rw_kin.json"
KIN_MODEL = "./models/rw_base.pth"

FR_CONFIG = "./configs/mlb_french.json"
FR_MODEL = "./models/"

VCTK_CONFIG = "./configs/vctk_base.json"
VCTK_MODEL = "./models/vctk.pth"

app = FastAPI()


eng_hps = utils.get_hparams_from_file(ENGLISH_CONFIG)
vctk_hps = utils.get_hparams_from_file(VCTK_CONFIG)
rw_hps = utils.get_hparams_from_file(KIN_CONFIG)
fr_hps = utils.get_hparams_from_file(FR_CONFIG)

#English -----------------------------------------------------
if GPU:
    net_g_gpu = SynthesizerTrn(
        len(symbols),
        eng_hps.data.filter_length // 2 + 1,
        eng_hps.train.segment_size // eng_hps.data.hop_length,
        **eng_hps.model
    ).cuda()
    _ = net_g_gpu.eval()
    _ = utils.load_checkpoint(ENGLISH_MODEL, net_g_gpu, None)



net_g = SynthesizerTrn(
    len(symbols),
    eng_hps.data.filter_length // 2 + 1,
    eng_hps.train.segment_size // eng_hps.data.hop_length,
    **eng_hps.model
).cpu()
_ = net_g.eval()
_ = utils.load_checkpoint(ENGLISH_MODEL, net_g, None)

#French -----------------------------------------------------

# if GPU:
#     fr_gpu = SynthesizerTrn(
#         len(fr_symbols),
#         fr_hps.data.filter_length // 2 + 1,
#         fr_hps.train.segment_size // fr_hps.data.hop_length,
#         **fr_hps.model
#     ).cuda()
#     _ = fr_gpu.eval()
#     _ = utils.load_checkpoint(FR_MODEL, fr_gpu, None)

# fr_cpu = SynthesizerTrn(
#     len(fr_symbols),
#     fr_hps.data.filter_length // 2 + 1,
#     fr_hps.train.segment_size // fr_hps.data.hop_length,
#     **fr_hps.model
# ).cpu()
# _ = net_g.eval()
# _ = utils.load_checkpoint(FR_MODEL, fr_cpu, None)

#Kinyarwanda -----------------------------------------------------

if GPU:
    rw_gpu = SynthesizerTrn(
        len(rw_symbols),
        rw_hps.data.filter_length // 2 + 1,
        rw_hps.train.segment_size // rw_hps.data.hop_length,
        **rw_hps.model
    ).cuda()
    _ = rw_gpu.eval()
    _ = utils.load_checkpoint(KIN_MODEL, rw_gpu, None)

rw_cpu = SynthesizerTrn(
    len(rw_symbols),
    rw_hps.data.filter_length // 2 + 1,
    rw_hps.train.segment_size // rw_hps.data.hop_length,
    **rw_hps.model
).cpu()
_ = rw_cpu.eval()
_ = utils.load_checkpoint(KIN_MODEL, rw_cpu, None)



#VCTK -----------------------------------------------------

if GPU:
    vctk_gpu_model = SynthesizerTrn(
        len(vctk_symbols),
        vctk_hps.data.filter_length // 2 + 1,
        vctk_hps.train.segment_size // vctk_hps.data.hop_length,
        n_speakers=vctk_hps.data.n_speakers,
        **vctk_hps.model).cuda()
    _ = vctk_gpu_model.eval()

    _ = utils.load_checkpoint("./models/vctk.pth", vctk_gpu_model, None)



vctk_cpu_model = SynthesizerTrn(
    len(vctk_symbols),
    vctk_hps.data.filter_length // 2 + 1,
    vctk_hps.train.segment_size // vctk_hps.data.hop_length,
    n_speakers=vctk_hps.data.n_speakers,
    **vctk_hps.model).cpu()
_ = vctk_cpu_model.eval()

_ = utils.load_checkpoint("./models/vctk.pth", vctk_cpu_model, None)


max_threads = os.cpu_count()


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    return torch.LongTensor(text_norm)

def get_text_vctk(text, hps):
    text_norm = vctk_text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_text_fr(text, hps):
    text_norm = fr_text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_text_rw(text, hps):
    text_norm = rw_text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

#Audio -----------------------------------------------------------------------------------

 



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

#French -----------------------------------------------------
# def fr_get_audio_gpu(stn_tst,net_g,hps):
#     with torch.no_grad():
#         x_tst = stn_tst.cuda().unsqueeze(0)
#         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
#         audio = fr_gpu.infer(
#             x_tst,
#             x_tst_lengths,
#             noise_scale=0.667,
#             noise_scale_w=0.8,
#             length_scale=1,
#         )[0][0, 0].data.cpu().float().numpy()

#     # Save audio to a file (temporary here, you can modify as needed)
#     audio_file = io.BytesIO()
#     write(audio_file, hps.data.sampling_rate, audio)
#     audio_file.seek(0)
#     return audio_file.read()

# def fr_get_audio_cpu(stn_tst,net_g,hps):
#     with torch.no_grad():
#         x_tst = stn_tst.cpu().unsqueeze(0)
#         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
#         audio = fr_cpu.infer(
#             x_tst,
#             x_tst_lengths,
#             noise_scale=0.667,
#             noise_scale_w=0.8,
#             length_scale=1,
#         )[0][0, 0].data.cpu().float().numpy()

#     audio_file = io.BytesIO()
#     write(audio_file, hps.data.sampling_rate, audio)
#     audio_file.seek(0)
#     return audio_file.read()


#Kinyarwanda -----------------------------------------------------

def rw_get_audio_gpu(stn_tst,rw_gpu,hps):
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = rw_gpu.infer(
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

def rw_get_audio_cpu(stn_tst,rw_cpu,hps):
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = rw_cpu.infer(
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

#VCTK ---------------------------------------------------------------------------

def vctk_gpu(stn_tst,vctk_gpu_model,hps):

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([4]).cuda()
        audio = vctk_gpu_model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

def vctk_cpu(stn_tst,vctk_cpu_model,hps):

    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        sid = torch.LongTensor([4]).cuda()
        audio = vctk_cpu_model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        
    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()






def get_audio(stn_tst,net_g_gpu,hps):
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g_gpu.infer(
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


    audio_files = []



    try:
        while True:
            data = await websocket.receive_text()


            file_data = data.replace("FILE:", "")
            file_lines = file_data.split("\n")
            print(len(file_lines))

            data = CustomData(file_lines, eng_hps)
            

            data_loader = data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = get_audio(stn_tst,net_g_gpu,eng_hps)
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

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()



@app.websocket("/english/cpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []



    try:
        while True:
            data = await websocket.receive_text()


            file_data = data.replace("FILE:", "")
            file_lines = file_data.split("\n")
            print(len(file_lines))

            data = CustomData(file_lines, eng_hps)
            

            data_loader = data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = get_audio_cpu(stn_tst,net_g,eng_hps)
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

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


@app.websocket("/rw/gpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []



    try:
        while True:
            data = await websocket.receive_text()


            file_data = data.replace("FILE:", "")
            file_lines = file_data.split("\n")
            print(len(file_lines))

            data = CustomData(file_lines, rw_hps)
            

            data_loader = data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = get_audio(stn_tst,rw_gpu,rw_hps)
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

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()



@app.websocket("/vctk/gpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []



    try:
        while True:
            data = await websocket.receive_text()


            file_data = data.replace("FILE:", "")
            file_lines = file_data.split("\n")
            print(len(file_lines))

            data = CustomData(file_lines, vctk_hps)
            

            data_loader = data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = get_audio_cpu(stn_tst,vctk_gpu_model,vctk_hps)
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

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

@app.websocket("/vctk/cpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []



    try:
        while True:
            data = await websocket.receive_text()


            file_data = data.replace("FILE:", "")
            file_lines = file_data.split("\n")
            print(len(file_lines))

            data = CustomData(file_lines, vctk_hps)
            

            data_loader = data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = get_audio_cpu(stn_tst,vctk_cpu_model,vctk_hps)
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

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
