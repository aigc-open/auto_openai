import time
import numpy as np
import librosa
from funasr import AutoModel
import gradio as gr
import torch
import os


class ASR:
    def __init__(self, device="cpu", root_path=".") -> None:
        self.model = AutoModel(model=os.path.join(root_path, "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
                               vad_model=os.path.join(
                                   root_path, "speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                               punc_model=os.path.join(
                                   root_path, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
                               device=device)

    def transcribe(self, sr, audio):
        audio = librosa.resample(audio,
                                 orig_sr=sr,
                                 target_sr=16000)
        res = self.model.generate(input=audio,
                                  batch_size_s=300,
                                  hotword='魔搭')
        if len(res) > 0:
            return res[0]["text"]
        return ""

    def transcribe_one(self, audio_path):
        audio, original_sample_rate = librosa.load(audio_path, sr=None)
        return self.transcribe(original_sample_rate, audio)


def run(port=7861, model_root_path="/workspace/code/gitlab/MuseTalk/models/"):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        device = "cuda"
    else:
        device = "cpu"
    asr = ASR(root_path=model_root_path, device=device)
    demo = gr.Interface(
        fn=asr.transcribe_one,
        inputs=gr.Audio(label="上传语音文件", type="filepath"),
        outputs=gr.Text(label="识别结果"),
        title="语音识别",
        description="上传语音文件进行识别"
    )

    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    from fire import Fire
    Fire(run)
