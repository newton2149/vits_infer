import asyncio
import websockets
import time

async def send_text_real_cpu():
    async with websockets.connect("ws://localhost:8000/english/cpu",timeout=10) as websocket:
        print("Realtime CPU Text to Speech")
        text = "This is a test sentence."
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"Send Text Runtime: {runtime} seconds")
        with open("output_audio_cpu.wav", "wb") as audio_file:
            audio_file.write(audio_data)

async def send_text_real_gpu():
    async with websockets.connect("ws://localhost:8000/english/gpu") as websocket:
        print("Realtime GPU Text to Speech")
        text = "This is a test sentence."
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"Send Text Runtime: {runtime} seconds")
        with open("output_audio_gpu.wav", "wb") as audio_file:
            audio_file.write(audio_data)

        
async def send_file_gpu():
    async with websockets.connect("ws://localhost:8001/english/gpu") as websocket:
        print("Batch GPU Text to Speech")
        file_path = "./test.txt"  
        with open(file_path, "r") as file:
            file_content = file.read()

        start = time.time()

        await websocket.send(f"FILE:{file_content}")

        audio_zip_data = await websocket.recv()


        with open("audio_files_gpu.zip", "wb") as audio_zip_file:
            audio_zip_file.write(audio_zip_data)

        end_time = time.time()
        runtime = end_time - start
        print(f"Send Text Runtime: {runtime} seconds")

        
async def send_file_cpu():
    async with websockets.connect("ws://localhost:8001/english/cpu") as websocket:
        print("Batch CPU Text to Speech")
        file_path = "./test.txt" 
        with open(file_path, "r") as file:
            file_content = file.read()

        start = time.time()

        await websocket.send(f"FILE:{file_content}")
        audio_zip_data = await websocket.recv()

        with open("audio_files_cpu.zip", "wb") as audio_zip_file:
            audio_zip_file.write(audio_zip_data)

        end_time = time.time()
        runtime = end_time - start
        print(f"Send Text Runtime: {runtime} seconds")




# asyncio.get_event_loop().run_until_complete(send_file_gpu())
# asyncio.get_event_loop().run_until_complete(send_file_cpu())

# asyncio.get_event_loop().run_until_complete(send_text_real_gpu())
asyncio.get_event_loop().run_until_complete(send_text_real_cpu())

