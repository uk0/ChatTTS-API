import io
import os
import random
import argparse
import struct

import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import ChatTTS

app = FastAPI()


def build_wav_header(sample_rate, num_samples, num_channels, bytes_per_sample):
    header = bytearray()
    header += b'RIFF'
    header += struct.pack('<L', 36 + num_samples * bytes_per_sample)
    header += b'WAVE'
    header += b'fmt '
    header += struct.pack('<L', 16)
    header += struct.pack('<H', 1)
    header += struct.pack('<H', num_channels)
    header += struct.pack('<L', sample_rate)
    header += struct.pack('<L', sample_rate * num_channels * bytes_per_sample)
    header += struct.pack('<H', num_channels * bytes_per_sample)
    header += struct.pack('<H', 8 * bytes_per_sample)
    header += b'data'
    header += struct.pack('<L', num_samples * bytes_per_sample)
    return header


def deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_audio(text, temperature, top_P, top_K, audio_seed_input, refine_text_flag):
    deterministic(audio_seed_input)
    rand_spk = chat.sample_random_speaker()
    params_infer_code = {
        'spk_emb': rand_spk,
        'temperature': temperature,
        'top_P': top_P,
        'top_K': top_K,
    }
    params_refine_text = {'prompt': '[oral_2][laugh_0][break_4]'}
    if refine_text_flag:
        text = chat.infer(text,
                          skip_refine_text=False,
                          refine_text_only=True,
                          params_refine_text=params_refine_text,
                          params_infer_code=params_infer_code
                          )

    wav = chat.infer(text,
                     skip_refine_text=True,
                     params_refine_text=params_refine_text,
                     params_infer_code=params_infer_code
                     )

    audio_data = np.array(wav[0]).flatten()
    sample_rate = 24000
    text_data = text[0] if isinstance(text, list) else text

    return [(sample_rate, audio_data), text_data]


@app.post("/api/tts")
async def tts_api(request: Request):
    data = await request.json()
    text = data.get("text", "")
    temperature = data.get("temperature", 0.1)
    top_P = data.get("top_P", 0.2)
    top_K = data.get("top_K", 10)
    audio_seed_input = data.get("audio_seed_input", 2)
    refine_text_flag = data.get("refine_text_flag", True)

    (sample_rate, audio_data), text_data = generate_audio(text, temperature, top_P, top_K, audio_seed_input,
                                                          refine_text_flag)

    # 构建WAV头
    num_samples = len(audio_data)
    num_channels = 1
    bytes_per_sample = 2
    wav_header = build_wav_header(sample_rate, num_samples, num_channels, bytes_per_sample)

    # 将音频数据转换为16位整数格式
    audio_data = (audio_data * 32767).astype(np.int16)

    # 创建字节流
    audio_bytes = io.BytesIO()

    # 写入WAV头和音频数据
    audio_bytes.write(wav_header)
    audio_bytes.write(audio_data.tobytes())
    audio_bytes.seek(0)

    # 返回音频文件流
    return StreamingResponse(audio_bytes, media_type="audio/wav")


def main():
    parser = argparse.ArgumentParser(description='ChatTTS API Launch')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server name')
    parser.add_argument('--server_port', type=int, default=8081, help='Server port')
    parser.add_argument('--local_path', type=str, default=None, help='the local_path if need')
    args = parser.parse_args()

    print("loading ChatTTS model...")
    global chat
    chat = ChatTTS.Chat()

    if args.local_path == None:
        chat.load_models()
    else:
        print('local model path:', args.local_path)
        chat.load_models('local', local_path=args.local_path)

    import uvicorn
    uvicorn.run(app, host=args.server_name, port=args.server_port)


if __name__ == '__main__':
    main()
