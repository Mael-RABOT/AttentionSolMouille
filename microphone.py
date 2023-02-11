import pyaudio
import wave
import time

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "file.wav"
TIME_BETWEEN_RECORDS = 5
NB_FILES = 5

def record_audio(wave_output_filename=WAVE_OUTPUT_FILENAME, nb=0):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(f"[{nb}] recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print(f"[{nb}] finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(wave_output_filename, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

for i in range(NB_FILES):
    pass
    #record_audio("file" + str(i + 1) + ".wav", i + 1)
    #time.sleep(TIME_BETWEEN_RECORDS)