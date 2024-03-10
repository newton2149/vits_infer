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
from serverutils import get_text, get_text_vctk, get_text_fr, get_text_rw, get_audio, get_audio_cpu, vctk_gpu, vctk_cpu, rw_get_audio_gpu, rw_get_audio_cpu, ENGLISH_MODEL, KIN_MODEL, FR_MODEL, VCTK_MODEL, eng_hps, vctk_hps, rw_hps, fr_hps, fr_get_audio_gpu, fr_get_audio_cpu
import random
from scipy.io.wavfile import write




GPU = torch.cuda.is_available()

def extract_zip(zip_file_path, extract_to='inference_db'):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:

        os.makedirs(extract_to, exist_ok=True)
        zip_ref.extractall(extract_to)

        return os.path.join(extract_to, zip_ref.namelist()[0])


def read_text_files(extract_to):
    for file in os.listdir(extract_to):
        with open(os.path.join(extract_to, file), 'r') as f:
            file_lines = f.readlines()
            return file_lines


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

if GPU:
    fr_gpu = SynthesizerTrn(
        len(fr_symbols),
        fr_hps.data.filter_length // 2 + 1,
        fr_hps.train.segment_size // fr_hps.data.hop_length,
        **fr_hps.model
    ).cuda()
    _ = fr_gpu.eval()
    _ = utils.load_checkpoint(FR_MODEL, fr_gpu, None)

fr_cpu = SynthesizerTrn(
    len(fr_symbols),
    fr_hps.data.filter_length // 2 + 1,
    fr_hps.train.segment_size // fr_hps.data.hop_length,
    **fr_hps.model
).cpu()
_ = net_g.eval()
_ = utils.load_checkpoint(FR_MODEL, fr_cpu, None)

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





@app.websocket("/english/ljspeech/gpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []
    try:
        while True:
            data = await websocket.receive_text()


            url_zip = data
            print(url_zip)
            
            #get zip file
            path = extract_zip(url_zip)
            file_lines = read_text_files(f"./{path}")
            print(file_lines)          
                    
            

            data = CustomData(file_lines, eng_hps,'en')
            

            data_loader  = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = get_audio(stn_tst,net_g_gpu,eng_hps)
                    audio_files.append(audio_data)                

            await websocket.send_text("Done Conversion")

            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        await websocket.close()



@app.websocket("/english/ljspeech/cpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []



    try:
        while True:
            data = await websocket.receive_text()


            url_zip = data
            print(url_zip)
            
            #get zip file
            path = extract_zip(url_zip)
            file_lines = read_text_files(f"./{path}")
            print(file_lines)

            data = CustomData(file_lines, eng_hps,'en')
            

            data_loader  = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = get_audio_cpu(stn_tst,net_g,eng_hps)
                    audio_files.append(audio_data)                

        
            await websocket.send_text("Done Conversion")
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


            url_zip = data
            print(url_zip)
            
            #get zip file
            path = extract_zip(url_zip)
            file_lines = read_text_files(f"./{path}")
            print(file_lines)

            data = CustomData(file_lines, rw_hps,'rw')
            

            data_loader  = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = rw_get_audio_gpu(stn_tst,rw_gpu,rw_hps)
                    audio_files.append(audio_data)
                    
            await websocket.send_text("Done Conversion")                



            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

@app.websocket("/rw/cpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()
    audio_files = []
    try:
        while True:
            data = await websocket.receive_text()


            url_zip = data
            print(url_zip)
            
            #get zip file
            path = extract_zip(url_zip)
            file_lines = read_text_files(f"./{path}")
            print(file_lines)

            data = CustomData(file_lines, rw_hps,'rw')
            

            data_loader  = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = rw_get_audio_cpu(stn_tst,rw_gpu,rw_hps)
                    audio_files.append(audio_data)
                    
            await websocket.send_text("Done Conversion")          



            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
        
        
@app.websocket("/french/cpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()
    audio_files = []

    try:
        while True:
            data = await websocket.receive_text()


            url_zip = data
            print(url_zip)
            
            #get zip file
            path = extract_zip(url_zip)
            file_lines = read_text_files(f"./{path}")
            print(file_lines)

            data = CustomData(file_lines, rw_hps,'fr')
            

            data_loader  = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = fr_get_audio_cpu(stn_tst,rw_gpu,rw_hps)
                    audio_files.append(audio_data)
                    
            await websocket.send_text("Done Conversion")                



            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
        
@app.websocket("/french/gpu")
async def text_to_audio(websocket: WebSocket):
    await websocket.accept()


    audio_files = []



    try:
        while True:
            data = await websocket.receive_text()


            url_zip = data
            print(url_zip)
            
            #get zip file
            path = extract_zip(url_zip)
            file_lines = read_text_files(f"./{path}")
            print(file_lines)

            data = CustomData(file_lines, rw_hps,'fr')
            

            data_loader  = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = fr_get_audio_gpu(stn_tst,rw_gpu,rw_hps)
                    audio_files.append(audio_data)
                    
            await websocket.send_text("Done Conversion")                



            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
        
        
@app.websocket("/english/vctk/gpu/{spk}/{noise_scale}")
async def text_to_audio(websocket: WebSocket ,  spk: int = 4, noise_scale: float = 0.667):
    await websocket.accept()


    audio_files = []

    try:
        while True:
            data = await websocket.receive_text()


            url_zip = data
            print(url_zip)
            
            #get zip file
            path = extract_zip(url_zip)
            file_lines = read_text_files(f"./{path}")
            print(file_lines)

            data = CustomData(file_lines, vctk_hps,'vctk')
            

            data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = vctk_gpu(stn_tst,vctk_gpu_model,vctk_hps)
                    audio_files.append(audio_data)                

            await websocket.send_text("Done Conversion")

            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

@app.websocket("/english/vctk/cpu/{spk}/{noise_scale}")
async def text_to_audio(websocket: WebSocket ,  spk: int = 4, noise_scale: float = 0.667):
    await websocket.accept()


    audio_files = []

    try:
        while True:
            data = await websocket.receive_text()


            url_zip = data
            print(url_zip)
            
            #get zip file
            path = extract_zip(url_zip)
            file_lines = read_text_files(f"./{path}")
            print(file_lines)

            data = CustomData(file_lines, vctk_hps,'vctk')
            

            data_loader = data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count(),)

            for id,stn_tst, in tqdm(data_loader):
                print(id)
                for stn_tst in stn_tst:
                    audio_data = vctk_cpu(stn_tst,vctk_gpu_model,vctk_hps)
                    write(f"./test-autio/{id}.wav",rate=vctk_hps.data.sampling_rate,data=audio_data)
                    audio_files.append(audio_data)                

            await websocket.send_text("Done Conversion")

            break

        
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()






if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
