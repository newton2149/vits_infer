import io
import torch
from fastapi import FastAPI, WebSocket
import utils
from models import SynthesizerTrn
from text.symbols import symbols

from text.mlb_fr_symbols import symbols as fr_symbols

from text.vctk_symbols import symbols as vctk_symbols

from text.rw_symbols import symbols as rw_symbols
import os
from tqdm import tqdm
from data import CustomData
import zipfile
from torch.utils.data import  DataLoader
from serverutils import get_text, get_text_vctk, get_text_fr, get_text_rw, get_audio, get_audio_cpu, vctk_gpu, vctk_cpu, rw_get_audio_gpu, rw_get_audio_cpu,ENGLISH_MODEL,KIN_MODEL,FR_MODEL,VCTK_MODEL,eng_hps,vctk_hps,rw_hps,fr_hps



GPU = torch.cuda.is_available()



app = FastAPI()



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
