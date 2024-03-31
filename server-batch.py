import io
import torch
from fastapi import FastAPI, File, HTTPException,Depends,BackgroundTasks
import utils
from models import SynthesizerTrn
from text.symbols import symbols
import psycopg2
from text.mlb_fr_symbols import symbols as fr_symbols
from text.vctk_symbols import symbols as vctk_symbols
from text.rw_symbols import symbols as rw_symbols
import os
from tqdm import tqdm
from data import CustomData
from torch.utils.data import DataLoader
from serverutils import (get_audio, get_audio_cpu, vctk_gpu, vctk_cpu, rw_get_audio_gpu, rw_get_audio_cpu,
                         ENGLISH_MODEL, KIN_MODEL, FR_MODEL, VCTK_MODEL, eng_hps, vctk_hps, rw_hps, fr_hps, fr_get_audio_gpu, fr_get_audio_cpu, download_file_from_firebase_storage, extract_zip, read_text_files, zip_wav_files)
import random
from dotenv import load_dotenv

from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker,declarative_base
from datetime import datetime
import uuid
import asyncio

load_dotenv()

cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIAL_FILE"))  
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")  
})


bucket = storage.bucket()
print("Bucket:", bucket)

# Database configuration
DB_NAME = os.getenv("DB_NAME")
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

# Create database if it doesn't exist
try:
    conn = psycopg2.connect(
        dbname="postgres",
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        host=os.getenv("POSTGRES_HOST"),
    )
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{DB_NAME}';")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f"CREATE DATABASE {DB_NAME};")
    cur.close()
    conn.close()
except psycopg2.Error as e:
    print("Error creating database:", e)

# Continue with FastAPI initialization
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    __tablename__ = "VITS_QUEUE"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model = Column(String)
    language = Column(String)
    data = Column(String)
    status = Column(String)
    result = Column(String, nullable=True)
    date = Column(String, default=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
        
        
def upload_file_to_firebase_storage(local_file_path, remote_file_path):
    try:
        from datetime import datetime, timedelta
        blob = bucket.blob(remote_file_path)
        blob.upload_from_filename(local_file_path)
        expiration_time = datetime.utcnow() + timedelta(hours=1)

        # Get the download URL
        download_url = blob.generate_signed_url(expiration=expiration_time)
        print("File uploaded successfully.")

        return download_url

    except Exception as e:
        print("Error uploading file:", e)
        


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

    _ = utils.load_checkpoint(VCTK_MODEL, vctk_gpu_model, None)


vctk_cpu_model = SynthesizerTrn(
    len(vctk_symbols),
    vctk_hps.data.filter_length // 2 + 1,
    vctk_hps.train.segment_size // vctk_hps.data.hop_length,
    n_speakers=vctk_hps.data.n_speakers,
    **vctk_hps.model).cpu()
_ = vctk_cpu_model.eval()

_ = utils.load_checkpoint(VCTK_MODEL, vctk_cpu_model, None)


max_threads = os.cpu_count()


async def english_ljspeech_gpu_h(text_request: TextRequest):
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
    url = upload_file_to_firebase_storage(zip_file_path, f"output-vits/{os.path.basename(zip_file_path)}")

    os.remove(local_zip)
    os.remove(zip_file_path)
    
    

    return url

def perform_ljspeech_gpu(job_id: String, data: TextRequest,db):
    response = asyncio.run(english_ljspeech_gpu_h(data))
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "COMPLETED", "result": response})
    db.commit()

    return {"job_id": job_id, "status": "COMPLETED"}


@app.post("/english/ljspeech/gpu")
async def english_ljspeech_gpu(data: TextRequest, background_tasks: BackgroundTasks,db=Depends(get_db)):
    
    data_string = data.model_dump_json()

    job = Job(data=data_string, status="PENDING", model="lJSpeech", language="ENGLISH")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(perform_ljspeech_gpu, job.id, data, db)

    return {"job_id": job.id, "status": job.status}
    

async def english_ljspeech_cpu_h(text_request: TextRequest):
    
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
    url = upload_file_to_firebase_storage(zip_file_path, f"output/{os.path.basename(zip_file_path)}")

    os.remove(local_zip)
    os.remove(zip_file_path)

    return url

def perform_ljspeech_cpu(job_id: String, data: TextRequest,db):
    response = asyncio.run(english_ljspeech_cpu_h(data))
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "COMPLETED", "result": response})
    db.commit()

    return {"job_id": job_id, "status": "COMPLETED"}

@app.post("/english/ljspeech/cpu")
async def english_ljspeech_cpu(data: TextRequest, background_tasks: BackgroundTasks,db=Depends(get_db)):
    
    data_string = data.model_dump_json()

    job = Job(data=data_string, status="PENDING", model="lJSpeech", language="ENGLISH")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(perform_ljspeech_cpu, job.id, data, db)

    return {"job_id": job.id, "status": job.status}


async def french_cpu_h(text_request: TextRequest):
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
    url = upload_file_to_firebase_storage(zip_file_path, f"output/{os.path.basename(zip_file_path)}")

    os.remove(local_zip)
    os.remove(zip_file_path)

    return url

def perform_french_cpu(job_id: String, data: TextRequest,db):
    response = asyncio.run(french_cpu_h(data))
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "COMPLETED", "result": response})
    db.commit()

    return {"job_id": job_id, "status": "COMPLETED"}

    

@app.post("/french/cpu")
async def french_cpu(data: TextRequest, background_tasks: BackgroundTasks,db=Depends(get_db)):
        
        data_string = data.model_dump_json()
    
        job = Job(data=data_string, status="PENDING", model="French", language="FRENCH")
        db.add(job)
        db.commit()
        db.refresh(job)
    
        background_tasks.add_task(perform_french_cpu, job.id, data, db)
    
        return {"job_id": job.id, "status": job.status}


async def french_gpu_h(text_request: TextRequest):
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
    url = upload_file_to_firebase_storage(zip_file_path, f"output/{os.path.basename(zip_file_path)}")

    os.remove(local_zip)
    os.remove(zip_file_path)

    return url

def perform_french_gpu(job_id: String, data: TextRequest,db):
    response = asyncio.run(french_gpu_h(data))
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "COMPLETED", "result": response})
    db.commit()

    return {"job_id": job_id, "status": "COMPLETED"}

@app.post("/french/gpu")
async def french_gpu(data: TextRequest, background_tasks: BackgroundTasks,db=Depends(get_db)   ):
        
        data_string = data.model_dump_json()
    
        job = Job(data=data_string, status="PENDING", model="French", language="FRENCH")
        db.add(job)
        db.commit()
        db.refresh(job)
    
        background_tasks.add_task(perform_french_gpu, job.id, data, db)
    
        return {"job_id": job.id, "status": job.status}


async def rw_cpu_h(text_request: TextRequest):
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
                stn, rw_cpu, rw_hps, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-audio-rw-cpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)
    url = upload_file_to_firebase_storage(zip_file_path, f"output/{os.path.basename(zip_file_path)}")

    os.remove(local_zip)
    os.remove(zip_file_path)

    return url

def perform_rw_cpu(job_id: String, data: TextRequest,db):
    response = asyncio.run(rw_cpu_h(data))
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "COMPLETED", "result": response})
    db.commit()

    return {"job_id": job_id, "status": "COMPLETED"}

@app.post("/rw/cpu")
async def rw_cpu(data: TextRequest, background_tasks: BackgroundTasks,db=Depends(get_db)):
        
        data_string = data.model_dump_json()
    
        job = Job(data=data_string, status="PENDING", model="Kinyarwanda", language="KINYARWANDA")
        db.add(job)
        db.commit()
        db.refresh(job)
    
        background_tasks.add_task(perform_rw_cpu, job.id, data, db)
    
        return {"job_id": job.id, "status": job.status}



async def rw_gpu_h(text_request: TextRequest):
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
    url = upload_file_to_firebase_storage(zip_file_path, f"output/{os.path.basename(zip_file_path)}")

    os.remove(local_zip)
    os.remove(zip_file_path)

    return url

def perform_rw_gpu(job_id: String, data: TextRequest,db):
    response = asyncio.run(rw_gpu_h(data))
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "COMPLETED", "result": response})
    db.commit()

    return {"job_id": job_id, "status": "COMPLETED"}

@app.post("/rw/gpu")
async def rw_gpu(data: TextRequest, background_tasks: BackgroundTasks,db=Depends(get_db)):
    data_string = data.model_dump_json()

    job = Job(data=data_string, status="PENDING", model="Kinyarwanda", language="KINYARWANDA")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(perform_rw_gpu, job.id, data, db)

    return {"job_id": job.id, "status": job.status}


async def english_vctk_cpu_h(text_request: TextRequestVctk):
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
            audio_data = vctk_cpu(stn, vctk_cpu_model, vctk_hps,
                                  speaker_id, noise_scale, noise_scale_w, length_scale)
            audio_files.append(audio_data)

    for id, audio in enumerate(audio_files):
        with open(f"./temp_db/generated/{id}.wav", "wb") as f:
            f.write(audio)

    zip_file_path = f"./temp_db/output/test-audio-vctk-cpu{random.randint(0,1000)}.zip"
    zip_wav_files("./temp_db/generated/", zip_file_path)
    url = upload_file_to_firebase_storage(zip_file_path, f"output/{os.path.basename(zip_file_path)}")

    os.remove(local_zip)
    os.remove(zip_file_path)

    return url

def perform_vctk_cpu(job_id: String, data: TextRequestVctk,db):
    response = asyncio.run(english_vctk_cpu_h(data))
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "COMPLETED", "result": response})
    db.commit()

    return {"job_id": job_id, "status": "COMPLETED"}



@app.post("/english/vctk/cpu")
async def english_vctk_cpu(data: TextRequestVctk, background_tasks: BackgroundTasks,db=Depends(get_db)):
    data_string = data.model_dump_json()

    job = Job(data=data_string, status="PENDING", model="VCTK", language="ENGLISH")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(perform_vctk_cpu, job.id, data, db)

    return {"job_id": job.id, "status": job.status}



async def english_vctk_gpu_h(text_request: TextRequestVctk):
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
    url = upload_file_to_firebase_storage(zip_file_path, f"output/{os.path.basename(zip_file_path)}")

    os.remove(local_zip)
    os.remove(zip_file_path)

    return url


def perform_vctk_gpu(job_id: String, data: TextRequestVctk,db):
    response = asyncio.run(english_vctk_gpu_h(data))
    db.query(Job).filter(Job.id == job_id).update(
        {"status": "COMPLETED", "result": response})
    db.commit()

    return {"job_id": job_id, "status": "COMPLETED"}


@app.post("/english/vctk/gpu")
async def english_vctk_gpu(data: TextRequestVctk, background_tasks: BackgroundTasks,db=Depends(get_db)):
    
    data_string = data.model_dump_json()

    job = Job(data=data_string, status="PENDING", model="VCTK", language="ENGLISH")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(perform_vctk_gpu, job.id, data, db)

    return {"job_id": job.id, "status": job.status}


@app.get("/jobs/{job_id}/")
async def get_job_status(job_id: str, db=Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"id": job.id, "data": job.data, "status": job.status, "result": job.result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
