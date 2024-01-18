import asyncio
import websockets
import time

sentence = "/home/navneeth/EgoPro/dnn/vits_infer/test_text.zip"
PORT = 8001

async def eng_send_text_real_cpu():
    async with websockets.connect(f"ws://localhost:{PORT}/english/ljspeech/cpu", timeout=10) as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        res = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"LJSpeech CPU Text Runtime: {runtime} seconds")
        # with open("eng_text_real_cpu.wav", "wb") as audio_file:
        #     audio_file.write(audio_data)
        
        print(res)



async def eng_send_text_real_gpu():
    async with websockets.connect(f"ws://localhost:{PORT}/english/ljspeech/gpu") as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        res = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"LJSpeech GPU Text Runtime: {runtime} seconds")
        print(res)



async def vctk_send_text_real_cpu():
    async with websockets.connect(f"ws://localhost:{PORT}/english/vctk/cpu/{4}/{0.667}", timeout=10) as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        res = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"VCTK CPU Text Runtime: {runtime} seconds")
        print(res)



async def vctk_send_text_real_gpu():
    async with websockets.connect(f"ws://localhost:{PORT}/english/vctk/gpu/{4}/{0.667}") as websocket:
        text = sentence
        start = time.time()
        await websocket.send(text)
        res = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(res)



async def fr_send_text_real_cpu():
    async with websockets.connect(f"ws://localhost:{PORT}/french/cpu", timeout=10) as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        res = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"French CPU Text Runtime: {runtime} seconds")
        print(res)
        # with open("fr_text_real_cpu.wav", "wb") as audio_file:



async def fr_send_text_real_gpu():

    async with websockets.connect(f"ws://localhost:{PORT}/french/gpu") as websocket:

        text = sentence

        start = time.time()
        await websocket.send(text)
        res = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(f"French GPU Text Runtime: {runtime} seconds")
        print(res)


async def rw_send_text_real_cpu():
    async with websockets.connect(f"ws://localhost:{PORT}/rw/cpu", timeout=10) as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        res = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(res)
        


async def rw_send_text_real_gpu():
    async with websockets.connect(f"ws://localhost:{PORT}/rw/gpu") as websocket:

        text = sentence
        start = time.time()
        await websocket.send(text)
        res = await websocket.recv()
        end_time = time.time()
        runtime = end_time - start
        print(res)


# asyncio.get_event_loop().run_until_complete(eng_send_text_real_cpu())
asyncio.get_event_loop().run_until_complete(eng_send_text_real_gpu())
# asyncio.get_event_loop().run_until_complete(vctk_send_text_real_cpu())
# asyncio.get_event_loop().run_until_complete(vctk_send_text_real_gpu())
# asyncio.get_event_loop().run_until_complete(fr_send_text_real_cpu())
# asyncio.get_event_loop().run_until_complete(fr_send_text_real_gpu())
# asyncio.get_event_loop().run_until_complete(rw_send_text_real_cpu())
# asyncio.get_event_loop().run_until_complete(rw_send_text_real_gpu())
