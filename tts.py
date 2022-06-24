import os
import torch
import sounddevice as sd
import time

device = torch.device('cpu')
torch.set_num_threads(4)
local_file = 'model.pt'
put_accent = True
put_yo = True

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

example_text = 'В недрах тундры выдры в г+етрах т+ырят в вёдра +ядра к+едров.'
sample_rate = 48000
speaker = 'kseniya' # aidar, eugene, baya, kseniya, xenia,random

# воспроизводим
def va_speak(what: str):
    audio = model.apply_tts(text=what+"..",
                                 speaker=speaker,
                                 sample_rate=sample_rate,
                                put_accent=put_accent,
                                put_yo=put_yo)
    sd.play(audio, sample_rate * 1.05)
    time.sleep((len(audio) / sample_rate) + 0.5)
    sd.stop()

va_speak(example_text)
