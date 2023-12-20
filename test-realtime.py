import asyncio
import websockets
import time

async def eng_send_text_real_cpu():
    async with websockets.connect("ws://localhost:8000/english/ljspeech/cpu",timeout=10) as websocket:

        text = "This is a test sentence."
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"LJSpeech CPU Text Runtime: {runtime} seconds")
        with open("eng_text_real_cpu.wav", "wb") as audio_file:
            audio_file.write(audio_data)

async def eng_send_text_real_gpu():
    async with websockets.connect("ws://localhost:8000/english/ljspeech/gpu") as websocket:

        text = "This is a test sentence."
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"LJSpeech GPU Text Runtime: {runtime} seconds")
        with open("eng_text_real_gpu.wav", "wb") as audio_file:
            audio_file.write(audio_data)

async def vctk_send_text_real_cpu():
    async with websockets.connect("ws://localhost:8000/english/vctk/cpu",timeout=10) as websocket:

        text = "This is a test sentence."
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"VCTK CPU Text Runtime: {runtime} seconds")
        with open("vctk_text_real_cpu.wav", "wb") as audio_file:
            audio_file.write(audio_data)

async def vctk_send_text_real_gpu():
    async with websockets.connect("ws://localhost:8000/english/vctk/gpu") as websocket:
        text = "This is a test sentence."
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"VCTK CPU Text Runtime: {runtime} seconds")
        with open("vctk_text_real_gpu.wav", "wb") as audio_file:
            audio_file.write(audio_data)
        
# async def fr_send_text_real_cpu():
#     async with websockets.connect("ws://localhost:8000/english/cpu",timeout=10) as websocket:
#         print("Realtime CPU Text to Speech")
#         text = "This is a test sentence."
#         start = time.time()
#         await websocket.send(text)
#         audio_data = await websocket.recv()
#         end_time = time.time()
#         runtime = end_time - start
#         print(f"Send Text Runtime: {runtime} seconds")
#         with open("output_audio_cpu.wav", "wb") as audio_file:
#             audio_file.write(audio_data)

# async def fr_send_text_real_gpu():
#     async with websockets.connect("ws://localhost:8000/english/gpu") as websocket:
#         print("Realtime GPU Text to Speech")
#         text = "This is a test sentence."
#         start = time.time()
#         await websocket.send(text)
#         audio_data = await websocket.recv()
#         end_time = time.time()
#         runtime = end_time - start
#         print(f"Send Text Runtime: {runtime} seconds")
#         with open("output_audio_gpu.wav", "wb") as audio_file:
#             audio_file.write(audio_data)

async def rw_send_text_real_cpu():
    async with websockets.connect("ws://localhost:8000/rw/cpu",timeout=10) as websocket:

        text = "This is a test sentence."
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"Kinyarwanda CPU Text Runtime: {runtime} seconds")
        with open("rw_text_real_cpu.wav", "wb") as audio_file:
            audio_file.write(audio_data)

async def rw_send_text_real_gpu():
    async with websockets.connect("ws://localhost:8000/rw/gpu") as websocket:

        text = "This is a test sentence."
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"Kinyarwanda GPU Text Runtime: {runtime} seconds")
        with open("rw_text_real_gpu.wav", "wb") as audio_file:
            audio_file.write(audio_data)





asyncio.get_event_loop().run_until_complete(eng_send_text_real_cpu())
asyncio.get_event_loop().run_until_complete(eng_send_text_real_gpu())
asyncio.get_event_loop().run_until_complete(vctk_send_text_real_cpu())
asyncio.get_event_loop().run_until_complete(vctk_send_text_real_gpu())
# asyncio.get_event_loop().run_until_complete(fr_send_text_real_cpu())
# asyncio.get_event_loop().run_until_complete(fr_send_text_real_gpu())
asyncio.get_event_loop().run_until_complete(rw_send_text_real_cpu())
asyncio.get_event_loop().run_until_complete(rw_send_text_real_gpu())




