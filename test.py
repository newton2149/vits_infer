import asyncio
import websockets

async def send_text():
    async with websockets.connect("ws://localhost:8000/english/ws/text") as websocket:
        text = "This is a test sentence."
        await websocket.send(text)
        audio_data = await websocket.recv()
        # Save audio data to a file
        with open("output_audio.wav", "wb") as audio_file:
            audio_file.write(audio_data)
        
async def send_file():
    async with websockets.connect("ws://localhost:8000/english/ws/text/gpu") as websocket:
        file_path = "./test.txt"  # Replace with the path to your text file
        with open(file_path, "r") as file:
            file_content = file.read()
        await websocket.send(f"FILE:{file_content}")
        audio_zip_data = await websocket.recv()
        # Save audio ZIP data to a file
        with open("audio_files_gpu.zip", "wb") as audio_zip_file:
            audio_zip_file.write(audio_zip_data)

        
async def send_file_cpu():
    async with websockets.connect("ws://localhost:8000/english/ws/text/cpu") as websocket:
        file_path = "./test.txt"  # Replace with the path to your text file
        with open(file_path, "r") as file:
            file_content = file.read()
        await websocket.send(f"FILE:{file_content}")
        audio_zip_data = await websocket.recv()
        # Save audio ZIP data to a file
        with open("audio_files_cpu.zip", "wb") as audio_zip_file:
            audio_zip_file.write(audio_zip_data)



# Test sending text
# asyncio.get_event_loop().run_until_complete(send_file())
asyncio.get_event_loop().run_until_complete(send_file_cpu())

# asyncio.get_event_loop().run_until_complete(send_text())
