import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import speech_recognition as sr
import soundfile as sf
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

model.to(device)
vocoder.to(device)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)




# Function to capture voice input from the user
def takeaudio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source,phrase_time_limit=5)

    with open('sample.wav', "wb") as f:
        f.write(audio.get_wav_data())


def resample(audio):
    audio_data, sr = librosa.load(audio, sr=None)
    resampled_audio = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
    sf.write(audio, resampled_audio, 16000, subtype='PCM_16')


def transcribe(audio):
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    waveform, sampling_rate = sf.read(audio)
    input_features = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription



def TTS(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    sf.write(f"audio.wav", speech.numpy(), samplerate=16000)

#takeaudio()
#resample('sample.wav')
text=transcribe('sample.wav')    
TTS(text)        

