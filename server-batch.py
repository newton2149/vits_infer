import io
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
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
from torch.utils.data import DataLoader
from serverutils import (get_audio, get_audio_cpu, vctk_gpu, vctk_cpu, rw_get_audio_gpu, rw_get_audio_cpu,
                         ENGLISH_MODEL, KIN_MODEL, FR_MODEL, VCTK_MODEL, eng_hps, vctk_hps, rw_hps, fr_hps, fr_get_audio_gpu, fr_get_audio_cpu)
import random
from scipy.io.wavfile import write
import requests
import numpy as np
from fastapi.responses import FileResponse
from pydantic import BaseModel


app = FastAPI(
    title="TTS Server",
    description="TTS Server for English, French and Kinyarwanda",
    version="0.1.0",



)
GPU = torch.cuda.is_available()
os.makedirs("temp_db", exist_ok=True)
os.makedirs("temp_db/generated", exist_ok=True)
os.makedirs("temp_db/output", exist_ok=True)


class TextRequest(BaseModel):
    url: str
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
    length_scale: float = 1


class TextRequestVctk(BaseModel):
    url: str
    noise_scale: float = 0.667
    noise_scale_w: float = 0.8
    length_scale: float = 1
    speaker_id: int = 0


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

# French -----------------------------------------------------

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

# Kinyarwanda -----------------------------------------------------

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


# VCTK -----------------------------------------------------

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


def zip_wav_files(input_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, input_dir)
                zipf.write(file_path, arcname=arcname)


def download_file_from_firebase_storage(download_url, output_file_path):
    response = requests.get(download_url)
    if response.status_code == 200:
        with open(output_file_path, 'wb') as f:
            f.write(response.content)
        print("File downloaded successfully.")
    else:
        raise HTTPException(status_code=response.status_code,
                            detail=f"Failed to download file. Status code: {response.status_code}")


@app.post("/english/ljspeech/gpu")
async def english_ljspeech_gpu(text_request: TextRequest):
    url_zip = text_request.url
    noise_scale = text_request.noise_scale
    noise_scale_w = text_request.noise_scale_w
    length_scale = text_request.length_scale

    local_zip = "./temp_db/ljspeech.zip"
    download_file_from_firebase_storage(url_zip, local_zip)

    path = extract_zip(local_zip)
    file_lines = read_text_files(f"./{path}")
    file_lines = [lines.strip() for lines in file_lines]

    data = CustomData(file_lines, eng_hps, 'en')
    data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

    audio_files = []
    for id, stn_tst in tqdm(data_loader):
        for stn in stn_tst:
            stn = stn[stn != -9999]
            audio_data = get_audio(
                stn, net_g_gpu, eng_hps, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-ljspeech-audio-cpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)

    os.remove(local_zip)

    return FileResponse(zip_file_path, media_type='application/zip')


@app.post("/english/ljspeech/cpu")
async def english_ljspeech_cpu(text_request: TextRequest):

    url_zip = text_request.url
    noise_scale = text_request.noise_scale
    noise_scale_w = text_request.noise_scale_w
    length_scale = text_request.length_scale

    local_zip = "./temp_db/ljspeech.zip"
    download_file_from_firebase_storage(url_zip, local_zip)

    path = extract_zip(local_zip)
    file_lines = read_text_files(f"./{path}")
    file_lines = [lines.strip() for lines in file_lines]

    data = CustomData(file_lines, eng_hps, 'en')
    data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

    audio_files = []
    for id, stn_tst in tqdm(data_loader):
        for stn in stn_tst:
            stn = stn[stn != -9999]
            audio_data = get_audio_cpu(
                stn, net_g, eng_hps, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/ljspeech-audio-gpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)

    os.remove(local_zip)

    return FileResponse(zip_file_path, media_type='application/zip')


@app.post("/french/cpu")
async def french_cpu(text_request: TextRequest):
    url_zip = text_request.url
    noise_scale = text_request.noise_scale
    noise_scale_w = text_request.noise_scale_w
    length_scale = text_request.length_scale

    local_zip = "./temp_db/french.zip"
    download_file_from_firebase_storage(url_zip, local_zip)

    path = extract_zip(local_zip)
    file_lines = read_text_files(f"./{path}")
    file_lines = [lines.strip() for lines in file_lines]

    data = CustomData(file_lines, fr_hps, 'fr')
    data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

    audio_files = []
    for id, stn_tst in tqdm(data_loader):
        for stn in stn_tst:
            stn = stn[stn != -9999]
            audio_data = fr_get_audio_cpu(
                stn, fr_cpu, fr_hps, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-audio-french-cpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)

    os.remove(local_zip)

    return FileResponse(zip_file_path, media_type='application/zip')


@app.post("/french/gpu")
async def french_gpu(text_request: TextRequest):
    url_zip = text_request.url
    noise_scale = text_request.noise_scale
    noise_scale_w = text_request.noise_scale_w
    length_scale = text_request.length_scale

    local_zip = "./temp_db/french.zip"
    download_file_from_firebase_storage(url_zip, local_zip)

    path = extract_zip(local_zip)
    file_lines = read_text_files(f"./{path}")
    file_lines = [lines.strip() for lines in file_lines]

    data = CustomData(file_lines, fr_hps, 'fr')
    data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

    audio_files = []
    for id, stn_tst in tqdm(data_loader):
        for stn in stn_tst:
            stn = stn[stn != -9999]
            audio_data = fr_get_audio_gpu(
                stn, fr_gpu, fr_hps, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-audio-french-gpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)

    os.remove(local_zip)

    return FileResponse(zip_file_path, media_type='application/zip')


@app.post("/rw/cpu")
async def rw_cpu(text_request: TextRequest):
    url_zip = text_request.url
    noise_scale = text_request.noise_scale
    noise_scale_w = text_request.noise_scale_w
    length_scale = text_request.length_scale

    local_zip = "./temp_db/RW.zip"
    download_file_from_firebase_storage(url_zip, local_zip)

    path = extract_zip(local_zip)
    file_lines = read_text_files(f"./{path}")
    file_lines = [lines.strip() for lines in file_lines]

    data = CustomData(file_lines, rw_hps, 'rw')
    data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

    audio_files = []
    for id, stn_tst in tqdm(data_loader):
        for stn in stn_tst:
            stn = stn[stn != -9999]
            audio_data = rw_get_audio_cpu(
                stn, rw_gpu, rw_hps, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-audio-rw-cpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)

    os.remove(local_zip)

    return FileResponse(zip_file_path, media_type='application/zip')


@app.post("/rw/gpu")
async def rw_gpu(text_request: TextRequest):
    url_zip = text_request.url
    noise_scale = text_request.noise_scale
    noise_scale_w = text_request.noise_scale_w
    length_scale = text_request.length_scale

    local_zip = "./temp_db/RW.zip"
    download_file_from_firebase_storage(url_zip, local_zip)

    path = extract_zip(local_zip)
    file_lines = read_text_files(f"./{path}")
    file_lines = [lines.strip() for lines in file_lines]

    data = CustomData(file_lines, rw_hps, 'rw')
    data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

    audio_files = []
    for id, stn_tst in tqdm(data_loader):
        for stn in stn_tst:
            stn = stn[stn != -9999]
            audio_data = rw_get_audio_gpu(
                stn, rw_gpu, rw_hps, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-audio-rw-gpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)

    os.remove(local_zip)

    return FileResponse(zip_file_path, media_type='application/zip')


@app.post("/english/vctk/cpu")
async def english_vctk_cpu(text_request: TextRequestVctk):
    url_zip = text_request.url
    noise_scale = text_request.noise_scale
    noise_scale_w = text_request.noise_scale_w
    length_scale = text_request.length_scale
    speaker_id = text_request.speaker_id

    local_zip = "./temp_db/vctk.zip"
    download_file_from_firebase_storage(url_zip, local_zip)

    path = extract_zip(local_zip)
    file_lines = read_text_files(f"./{path}")
    file_lines = [lines.strip() for lines in file_lines]

    data = CustomData(file_lines, vctk_hps, 'vctk')
    data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

    audio_files = []
    for id, stn_tst in tqdm(data_loader):
        for stn in stn_tst:
            stn = stn[stn != -9999]
            audio_data = vctk_cpu(stn, vctk_gpu_model, vctk_hps,
                                  speaker_id, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-audio-vctk-cpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)

    os.remove(local_zip)

    return FileResponse(zip_file_path, media_type='application/zip')


@app.post("/english/vctk/gpu")
async def english_vctk_gpu(text_request: TextRequestVctk):
    url_zip = text_request.url

    noise_scale = text_request.noise_scale
    noise_scale_w = text_request.noise_scale_w
    length_scale = text_request.length_scale
    speaker_id = text_request.speaker_id

    local_zip = "./temp_db/vctk.zip"
    download_file_from_firebase_storage(url_zip, local_zip)

    path = extract_zip(local_zip)
    file_lines = read_text_files(f"./{path}")
    file_lines = [lines.strip() for lines in file_lines]

    data = CustomData(file_lines, vctk_hps, 'vctk')
    data_loader = DataLoader(data, batch_size=32, num_workers=os.cpu_count())

    audio_files = []
    for id, stn_tst in tqdm(data_loader):
        for stn in stn_tst:
            stn = stn[stn != -9999]
            audio_data = vctk_gpu(stn, vctk_gpu_model, vctk_hps,
                                  speaker_id, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-audio-vctk-gpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)

    os.remove(local_zip)

    return FileResponse(zip_file_path, media_type='application/zip')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
