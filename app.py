import streamlit as st
import asyncio
import websockets
import time
import random
from scipy.io.wavfile import write
import numpy as np

audio = None

async def eng_send_text_real_cpu(text):
    async with websockets.connect("ws://localhost:8000/english/ljspeech/cpu",timeout=10) as websocket:

        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"LJSpeech CPU Text Runtime: {runtime} seconds")
        return audio_data

async def fr_send_text_real_gpu(text):
    async with websockets.connect("ws://localhost:8000/french/gpu") as websocket:

        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"MLB GPU Text Runtime: {runtime} seconds")
        return audio_data
    
async def fre_send_text_real_cpu(text):
    async with websockets.connect("ws://localhost:8000/french/cpu",timeout=10) as websocket:

        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"MLB CPU Text Runtime: {runtime} seconds")
        return audio_data

async def eng_send_text_real_gpu(text):
    async with websockets.connect("ws://localhost:8000/english/ljspeech/gpu") as websocket:

        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"LJSpeech GPU Text Runtime: {runtime} seconds")
        return audio_data
async def vctk_send_text_real_cpu(text,spk,noise_scale):
    async with websockets.connect(f"ws://localhost:8000/english/vctk/cpu/{spk}/{noise_scale}",timeout=10) as websocket:

        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"VCTK CPU Text Runtime: {runtime} seconds")
        return audio_data

async def vctk_send_text_real_gpu(text,spk,noise_scale):
    async with websockets.connect(f"ws://localhost:8000/english/vctk/gpu/{spk}/{noise_scale}") as websocket:
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"VCTK GPU Text Runtime: {runtime} seconds")
        return audio_data
            
async def rw_send_text_real_cpu(text):
    async with websockets.connect("ws://localhost:8000/rw/cpu",timeout=10) as websocket:

        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"Kinyarwanda CPU Text Runtime: {runtime} seconds")
        return audio_data

async def rw_send_text_real_gpu(text):
    async with websockets.connect("ws://localhost:8000/rw/gpu") as websocket:

        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"Kinyarwanda GPU Text Runtime: {runtime} seconds")
        return audio_data


def getInference(model,device):
    
    if model == 'English' and device == 'CPU':
        return eng_send_text_real_cpu
    
    if model == 'English' and device == 'GPU':
        return eng_send_text_real_gpu
    
    if model == 'French' and device == 'CPU':
        return fre_send_text_real_cpu
    
    if model == 'French' and device == 'GPU':
        return fr_send_text_real_gpu
    
    
    
    if model == 'Kinyarwanda' and device == 'CPU':
        return rw_send_text_real_cpu
    
    if model == 'Kinyarwanda' and device == 'GPU':
        return rw_send_text_real_gpu
    
    





def main():
    
    audio = None

    st.set_page_config(layout='wide', page_title='VITS Testing Dashboard')

    models = ['English','Kinyarwanda','VCTK','French']
    de = ['CPU','GPU']


    st.write('''
    # VITS Testing Dashboard''')


    col1, col2 = st.columns(2)

    with col1:
        
        st.write('''
        ### Input
        ''')

        row, col = st.columns(2)

        dropdown = st.selectbox('Select Model', models)
        device = st.selectbox('Select Device', de)  
        
        if dropdown == 'VCTK':
            
            spk = st.number_input(label = 'Speaker ID',help = 'Enter the speaker ID',min_value=0,max_value=108,value=0,step=1)  
            noise_scale = st.number_input(label = 'Noise Scale',help = 'Enter the noise scale',min_value=0.0,max_value=1.0,value=0.667,step=0.1)          


        text = st.text_area(label = 'Text',help = 'Enter the name of the test')


        if st.button('Inference'):
            
            if dropdown == 'VCTK' and device == 'CPU':
                audio = asyncio.run(vctk_send_text_real_cpu(text,spk,noise_scale))
            elif dropdown == 'VCTK' and device == 'GPU':
                audio = asyncio.run(vctk_send_text_real_gpu(text,spk,noise_scale))
            
            else:
                audio = asyncio.run(getInference(dropdown,device)(text))
            
        
        

    # with col2:
    #     st.write('### Batch Inference')
    #     uploaded_files = st.file_uploader("Choose a Text File")

    #     if uploaded_files is not None:
    #         bytes_data = uploaded_files.getvalue()
    #         st.write(bytes_data)
                
    #         print(uploaded_files)
    #     if st.button('Inference Batch'):
    #         output = send_file(bytes_data)


        
    #     if output is not None:
    #         st.download_button(label='Download Audio',data=audio,file_name='audio.mp3',mime='audio/mp3')
        

        


    with col2:
        st.write('''
        ### Output
        ''')


        st.audio(audio,format='audio/mp3',sample_rate=22050)
        if audio is not None:
            st.download_button(label='Download Audio',data=audio,file_name=f'{random.randint(1,1000)}.wav',mime='audio/mp3')
        
    

    


if __name__ == "__main__":
    main()