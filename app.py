import streamlit as st
import IPython.display as ipd
import os
import torch
from torch import nn
from torch.nn import functional as F
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write
import asyncio
import websockets




CONFIG_DIR = "configs"
config = ""
audio = None
output = None

ENGLISH_TXT = 'ws://0.0.0.0:8000/english/ws/text'
ENGLISH_ZIP = 'ws://localhost:8000/english/ws/zip'



async def send_text():
    async with websockets.connect("ws://localhost:8000/english/ws/text") as websocket:
        text = "This is a test sentence."
        await websocket.send(text)
        audio_data = await websocket.recv()
        # Save audio data to a file
        with open("output_audio.wav", "wb") as audio_file:
            audio_file.write(audio_data)
        
async def send_file(file_content):
    async with websockets.connect("ws://localhost:8000/english/ws/text") as websocket:
        # file_path = "./test.txt"  # Replace with the path to your text file
        # with open(file_path, "r") as file:
        #     file_content = file.read()
        await websocket.send(f"FILE:{file_content}")
        audio_zip_data = await websocket.recv()
        # Save audio ZIP data to a file
        with open("audio_files.zip", "wb") as audio_zip_file:
            audio_zip_file.write(audio_zip_data)




def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm








def inference(text):
    return asyncio.run(send_text_and_receive_audio(text))

 

def inference_file():


   pass




st.set_page_config(layout='wide', page_title='VITS Testing Dashboard')

MODEL_PATH = '/home/navneeth/EgoPro/dnn/vits_kinyarwanda/models'

models = ['English','Kinyarwanda']

st.write('''
# VITS Testing Dashboard''')

col1, col2 = st.columns(2)

with col1:
    st.write('''
    ### Input
    ''')

    row, col = st.columns(2)

    dropdown = st.selectbox('Select Model', models)
    if dropdown == 'English':
        config = 'ljs_base.json'
        model = '/home/navneeth/EgoPro/dnn/vits_infer/models/ljspeech.pth'

    elif dropdown == 'Kinyarwanda':
        config = 'rw_kin.json'
        model = '/home/navneeth/EgoPro/dnn/vits_infer/models/ljspeech.pth'


    


    


    text = st.text_area(label = 'Text',help = 'Enter the name of the test')


    if st.button('Inference'):
        audio = inference(text)

with col2:
    st.write('### Batch Inference')
    uploaded_files = st.file_uploader("Choose a Text File")

    if uploaded_files is not None:
        bytes_data = uploaded_files.getvalue()
        st.write(bytes_data)
            
        print(uploaded_files)
    if st.button('Inference Batch'):
        output = send_file(bytes_data)


    
    if output is not None:
        st.download_button(label='Download Audio',data=audio,file_name='audio.mp3',mime='audio/mp3')
    

    



st.write('''
### Output
''')


st.audio(audio,format='audio/mp3',sample_rate=22050)
    

    


