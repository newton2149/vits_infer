import asyncio
import websockets
import time

sentence = """In the vast expanse of the cosmos, where stars twinkle like distant diamonds against the velvety backdrop of space, an intrepid explorer. """


async def eng_send_text_real_cpu():
    async with websockets.connect("ws://localhost:8000/english/ljspeech/cpu", timeout=10) as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        # print(f"LJSpeech CPU Text Runtime: {runtime} seconds")
        # with open("eng_text_real_cpu.wav", "wb") as audio_file:
        #     audio_file.write(audio_data)

        return audio_data


async def eng_send_text_real_gpu():
    async with websockets.connect("ws://localhost:8000/english/ljspeech/gpu") as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        # print(f"LJSpeech GPU Text Runtime: {runtime} seconds")
        # with open("eng_text_real_gpu.wav", "wb") as audio_file:
        #     audio_file.write(audio_data)


async def vctk_send_text_real_cpu():
    async with websockets.connect(f"ws://localhost:8000/english/vctk/cpu/{4}/{0.667}", timeout=10) as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        # print(f"VCTK CPU Text Runtime: {runtime} seconds")
        # with open("vctk_text_real_cpu.wav", "wb") as audio_file:
        #     audio_file.write(audio_data)


async def vctk_send_text_real_gpu():
    async with websockets.connect(f"ws://localhost:8000/english/vctk/gpu/{4}/{0.667}") as websocket:
        text = sentence
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        # print(f"VCTK CPU Text Runtime: {runtime} seconds")
        # with open("vctk_text_real_gpu.wav", "wb") as audio_file:
        #     audio_file.write(audio_data)


async def fr_send_text_real_cpu():
    async with websockets.connect("ws://localhost:8000/french/cpu", timeout=10) as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        # print(f"French CPU Text Runtime: {runtime} seconds")
        # with open("fr_text_real_cpu.wav", "wb") as audio_file:
        #     audio_file.write(audio_data)


async def fr_send_text_real_gpu():

    text = "This is a test sentence."
    async with websockets.connect("ws://localhost:8000/french/gpu") as websocket:

        text = sentence

        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        # print(f"French GPU Text Runtime: {runtime} seconds")


async def rw_send_text_real_cpu():
    async with websockets.connect("ws://localhost:8000/rw/cpu", timeout=10) as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        # print(f"Kinyarwanda CPU Text Runtime: {runtime} seconds")
        # with open("rw_text_real_cpu.wav", "wb") as audio_file:
        #     audio_file.write(audio_data)


async def rw_send_text_real_gpu():
    async with websockets.connect("ws://localhost:8000/rw/gpu") as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        audio_data = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        # print(f"Kinyarwanda GPU Text Runtime: {runtime} seconds")
        # with open("rw_text_real_gpu.wav", "wb") as audio_file:
        #     audio_file.write(audio_data)


# audio = asyncio.get_event_loop().run_until_complete(eng_send_text_real_cpu())
# with open("eng_text_real_gpu.wav", "wb") as audio_file:
#             audio_file.write(audio)

# asyncio.get_event_loop().run_until_complete(eng_send_text_real_cpu())
# asyncio.get_event_loop().run_until_complete(eng_send_text_real_gpu())
# asyncio.get_event_loop().run_until_complete(vctk_send_text_real_cpu())
# asyncio.get_event_loop().run_until_complete(vctk_send_text_real_gpu())
# asyncio.get_event_loop().run_until_complete(fr_send_text_real_cpu())
asyncio.get_event_loop().run_until_complete(fr_send_text_real_gpu())
# asyncio.get_event_loop().run_until_complete(rw_send_text_real_cpu())
# asyncio.get_event_loop().run_until_complete(rw_send_text_real_gpu())
