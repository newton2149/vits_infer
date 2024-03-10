import io
import torch
from text import text_to_sequence

from scipy.io.wavfile import write
import utils

from text import text_to_sequence

from text.mlb_fr import text_to_sequence as fr_text_to_sequence


from text.vctk import text_to_sequence as vctk_text_to_sequence

from text.rw import text_to_sequence as rw_text_to_sequence


from scipy.io.wavfile import write
import commons

ENGLISH_CONFIG = "./configs/ljs_base.json"
ENGLISH_MODEL = "./models/ljspeech.pth"

KIN_CONFIG = "./configs/rw_kin.json"
KIN_MODEL = "./models/rw_base.pth"

FR_CONFIG = "./configs/mlb_french.json"
FR_MODEL = "./models/french.pth"

VCTK_CONFIG = "./configs/vctk_base.json"
VCTK_MODEL = "./models/vctk.pth"


eng_hps = utils.get_hparams_from_file(ENGLISH_CONFIG)
vctk_hps = utils.get_hparams_from_file(VCTK_CONFIG)
rw_hps = utils.get_hparams_from_file(KIN_CONFIG)
fr_hps = utils.get_hparams_from_file(FR_CONFIG)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_text_vctk(text, hps):
    text_norm = vctk_text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_text_fr(text, hps):
    text_norm = fr_text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def get_text_rw(text, hps):
    text_norm = rw_text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

#Audio -----------------------------------------------------------------------------------

 



def get_audio(stn_tst,net_g,hps):
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0].data.cpu().float().numpy()

    # Save audio to a file (temporary here, you can modify as needed)
    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

def get_audio_cpu(stn_tst,net_g,hps):
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = net_g.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0].data.cpu().float().numpy()

    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

#French -----------------------------------------------------
def fr_get_audio_gpu(stn_tst,fr_gpu,hps):
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = fr_gpu.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0].data.cpu().float().numpy()

    # Save audio to a file (temporary here, you can modify as needed)
    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

def fr_get_audio_cpu(stn_tst,fr_cpu,hps):
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = fr_cpu.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0].data.cpu().float().numpy()

    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()


#Kinyarwanda -----------------------------------------------------

def rw_get_audio_gpu(stn_tst,rw_gpu,hps):
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = rw_gpu.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0].data.cpu().float().numpy()

    # Save audio to a file (temporary here, you can modify as needed)
    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

def rw_get_audio_cpu(stn_tst,rw_cpu,hps):
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = rw_cpu.infer(
            x_tst,
            x_tst_lengths,
            noise_scale=0.667,
            noise_scale_w=0.8,
            length_scale=1,
        )[0][0, 0].data.cpu().float().numpy()

    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

#VCTK ---------------------------------------------------------------------------

def vctk_gpu(stn_tst,vctk_gpu_model,hps,spk=4,noise_scale=0.667):

    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([spk]).cuda()
        audio = vctk_gpu_model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

def vctk_cpu(stn_tst,vctk_cpu_model,hps,spk=4,noise_scale=0.667):

    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        sid = torch.LongTensor([spk]).cpu()
        audio = vctk_cpu_model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        
    audio_file = io.BytesIO()
    write(audio_file, hps.data.sampling_rate, audio)
    audio_file.seek(0)
    return audio_file.read()

