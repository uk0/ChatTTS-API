
import requests
import argparse
import io
from scipy.io import wavfile

def send_tts_request(url, text, temperature=0.3, top_P=0.7, top_K=20, audio_seed_input=4099, text_seed_input=4099, refine_text_flag=True):
    data = {
        "text": text,
        "temperature": temperature,
        "top_P": top_P,
        "top_K": top_K,
        "audio_seed_input": audio_seed_input,
        "text_seed_input": text_seed_input,
        "refine_text_flag": refine_text_flag
    }
    print(f"Request data: {data}")

    response = requests.post(url, json=data)

    if response.status_code == 200:
        content_type = response.headers.get("Content-Type")
        if content_type == "audio/wav":
            audio_data = response.content
            return audio_data
        else:
            print(f"Unexpected content type: {content_type}")
    else:
        print(f"Request failed with status code: {response.status_code}")

    return None

def save_audio(audio_data, output_path):
    audio_stream = io.BytesIO(audio_data)
    sample_rate, audio_numpy = wavfile.read(audio_stream)
    wavfile.write(output_path, sample_rate, audio_numpy)
    print(f"Audio saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatTTS API Demo Client")
    parser.add_argument("--url", type=str, default="http://localhost:8081/api/tts", help="API URL")
    parser.add_argument("--text", type=str,default='红是朱砂痣烙印心口 红是蚊子血般平庸时间美化那仅有的悸动 也磨平激动 从背后抱你的时候 期待的却是她的面容 说来实在嘲讽 我不太懂偏渴望你懂 是否幸福轻得太沉重 过度使用不痒不痛 烂熟透红空洞了的瞳孔', help="Input text")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
    args = parser.parse_args()

    audio_data = send_tts_request(args.url, args.text)

    if audio_data:
        save_audio(audio_data, args.output)