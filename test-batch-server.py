import requests
from fastapi.responses import FileResponse
base_url = "http://localhost:8001"

english = "https://firebasestorage.googleapis.com/v0/b/intenrship-e76b3.appspot.com/o/test-zip.zip?alt=media&token=2279f00f-0007-4895-b891-3654770c4631"
french = "https://firebasestorage.googleapis.com/v0/b/intenrship-e76b3.appspot.com/o/french_text.zip?alt=media&token=e875874d-c427-49b6-bd1e-81c446932755"
kinyarwanda = "https://firebasestorage.googleapis.com/v0/b/intenrship-e76b3.appspot.com/o/kinyarwanda_text.zip?alt=media&token=ed1526cc-c57a-45bc-acd9-16199f376fcf"

def test_english_cpu()-> FileResponse:
    url = f"{base_url}/english/ljspeech/cpu"
    data = {
        "url":english,
        "noise_scale":0.667,
        "noise_scale_w":0.6,
        "length_scale":1,
        
    }
    response = requests.post(url, json=data)
    return response


def test_english_gpu()-> FileResponse:
    url = f"{base_url}/english/ljspeech/gpu"
    data = {
        "url":english,
        "noise_scale":0.667,
        "noise_scale_w":0.6,
        "length_scale":1,
        
    }
    response = requests.post(url, json=data)
    return response


def test_vctk_cpu()-> FileResponse:
    url = f"{base_url}/english/vctk/cpu"
    data = {
        "url":english,
        "speaker_id":4,
        "noise_scale":0.667,
        "noise_scale_w":0.6,
        "length_scale":1,
        
    }
    response = requests.post(url, json=data)
    return response

def test_vctk_gpu()-> FileResponse:
    url = f"{base_url}/english/vctk/gpu"
    data = {
        "url":english,
        "speaker_id":4,
        "noise_scale":0.667,
        "noise_scale_w":0.6,
        "length_scale":1,
        
    }
    response = requests.post(url, json=data)
    return response

def test_fr_gpu():
    url = f"{base_url}/french/gpu"
    data = {
        "url":french,
        "noise_scale":0.667,
        "noise_scale_w":0.6,
        "length_scale":1,
        
    }
    response = requests.post(url, json=data)
    return response

def test_fr_cpu():
    url = f"{base_url}/french/gpu"
    data = {
        "url":french,
        "noise_scale":0.667,
        "noise_scale_w":0.6,
        "length_scale":1,
        
    }
    response = requests.post(url, json=data)
    return response


def test_rw_gpu():
    url = f"{base_url}/rw/gpu"
    data = {
        "url":kinyarwanda,
        "noise_scale":0.667,
        "noise_scale_w":0.6,
        "length_scale":1,
        
    }
    response = requests.post(url, json=data)
    return response


def test_rw_cpu():
    url = f"{base_url}/rw/cpu"
    data = {
        "url":kinyarwanda,
        "noise_scale":0.667,
        "noise_scale_w":0.6,
        "length_scale":1,
        
    }
    response = requests.post(url, json=data)
    return response


if __name__ == "__main__":
    response_cpu = test_english_cpu()
    response_gpu = test_english_gpu()
    response_vctk = test_vctk_cpu()
    response_vctk = test_vctk_gpu()
    
    print(response_cpu)
    print(response_gpu)
    print(response_vctk)
    
    # print(response_fr_cpu)
    