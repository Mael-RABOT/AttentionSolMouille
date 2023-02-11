import pyaudio
import wave
import time
from tqdm import tqdm

FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 2
FOLDER = "assets/"
FILENAME = "file"
TIME_BETWEEN_RECORDS = 5
NB_FILES = 5

def record_audio(wave_output_filename, nb, nb_files, record_seconds):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(f"[{nb}/{nb_files}] recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print(f"[{nb}/{nb_files}] finished recording")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(wave_output_filename, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

def start_record(nb_files=NB_FILES, record_seconds=RECORD_SECONDS, time_between_records=TIME_BETWEEN_RECORDS):
    if nb_files == 0:
        print("Nothing to do")
        return
    for i in tqdm(range(nb_files)):
        record_audio(FOLDER + FILENAME + str(i + 1) + ".wav", i + 1, nb_files, record_seconds)
        if time_between_records != 0 and i < nb_files - 1:
            print("wait...")
            time.sleep(time_between_records)
    print("Done!\n")
    print("Files created:")
    for i in range(nb_files):
        print(f"{FILENAME}{str(i + 1)}.wav")

start_record(NB_FILES, RECORD_SECONDS, TIME_BETWEEN_RECORDS)
