# extract fbanck from wav and save to file
# pre processd an audio in 0.09912s

import os
from glob import glob
from python_speech_features import fbank
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool
from pydub import AudioSegment

import src.silence_detector as silence_detector
import src.constants as c
from src.constants import SAMPLE_RATE
from time import time

np.set_printoptions(threshold=0)
#pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', 100)


def find_files(directory, pattern='**/*.wav'):
    """Recursively finds all files matching the pattern."""
    return glob(os.path.join(directory, pattern), recursive=True)


def cvt_flac2wav(input_file, output_dir):
    inputext = input_file.replace('\\', '/')
    inputext = inputext.split('/')[-1].split('.')[0]
    output_path = os.path.join(
        output_dir,
        inputext +
        ".wav")

    if os.path.exists(output_path):
        print("File already exists: ", output_path)
        return output_path

    else:
        audio = AudioSegment.from_file(input_file, format="flac")
        audio.export(output_path, format="wav")
        print("Converted: ", output_path)
        return output_path


def cvt_process_and_save(
        input_dir=c.WAV_DIR,
        output_dir=c.WAV_DIR,
        num_workers=4):
    files = find_files(directory=input_dir, pattern='**/*.flac')
    with Pool(num_workers) as pool:
        pool.starmap(cvt_flac2wav, [(file, output_dir) for file in files])


def VAD(audio):
    chunk_size = int(SAMPLE_RATE*0.05) # 50ms
    index = 0
    sil_detector = silence_detector.SilenceDetector(15)
    nonsil_audio=[]
    while index + chunk_size < len(audio):
        if not sil_detector.is_silence(audio[index: index+chunk_size]):
            nonsil_audio.extend(audio[index: index + chunk_size])
        index += chunk_size

    return np.array(nonsil_audio)


def add_noise(main_audio, noise):
    # Adjust the amplitude of the secondary audio
    amplitude_factor = np.random.uniform(0.1, 0.3)
    noise *= amplitude_factor

    # Ensure the secondary audio is the same length as the main audio
    if len(noise) > len(main_audio):
        noise = noise[:len(main_audio)]
    else:
        noise = np.pad(noise, (0, len(main_audio) - len(noise)), 'constant')

    # Combine the main and secondary audio
    combined_audio = main_audio + noise
    return combined_audio


def read_audio(filename, noisename=None, sample_rate=SAMPLE_RATE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    if c.ADD_NOISE:
        noise, _ = librosa.load(noisename, sr=sample_rate, mono=True)
        audio = add_noise(audio, noise)
    audio = VAD(audio.flatten())
    start_sec, end_sec = c.TRUNCATE_SOUND_SECONDS
    start_frame = int(start_sec * SAMPLE_RATE)
    end_frame = int(end_sec * SAMPLE_RATE)

    if len(audio) < (end_frame - start_frame):
        au = [0] * (end_frame - start_frame)
        for i in range(len(audio)):
            au[i] = audio[i]
        audio = np.array(au)
    return audio

def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]


def extract_features(signal=np.random.uniform(size=48000), target_sample_rate=SAMPLE_RATE):
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64, winlen=0.025)   #filter_bank (num_frames , 64),energies (num_frames ,)
    #delta_1 = delta(filter_banks, N=1)
    #delta_2 = delta(delta_1, N=1)

    filter_banks = normalize_frames(filter_banks)
    #delta_1 = normalize_frames(delta_1)
    #delta_2 = normalize_frames(delta_2)

    #frames_features = np.hstack([filter_banks, delta_1, delta_2])    # (num_frames , 192)
    frames_features = filter_banks     # (num_frames , 64)
    num_frames = len(frames_features)
    return np.reshape(np.array(frames_features),(num_frames, 64, 1))   #(num_frames,64, 1)

def data_catalog(dataset_dir=c.DATASET_DIR, pattern='*.npy'):
    libri = pd.DataFrame()
    libri['filename'] = find_files(dataset_dir, pattern=pattern)
    libri['filename'] = libri['filename'].apply(lambda x: x.replace('\\', '/'))  # normalize windows paths
    libri['speaker_id'] = libri['filename'].apply(lambda x: x.split('/')[-1].split('-')[0])
    num_speakers = len(libri['speaker_id'].unique())
    print('Found {} files with {} different speakers.'.format(str(len(libri)).zfill(7), str(num_speakers).zfill(5)))
    # print(libri.head(10))
    return libri

def prep(libri,out_dir=c.DATASET_DIR,name='0', noisefiles=None):
    start_time = time()
    i=0
    for i in range(len(libri)):
        orig_time = time()
        filename = libri[i:i+1]['filename'].values[0]
        target_filename = out_dir + filename.split("/")[-1].split('.')[0] + '.npy'
        if os.path.exists(target_filename):
            if i % 10 == 0: print("task:{0} No.:{1} Exist File:{2}".format(name, i, filename))
            continue
        if c.ADD_NOISE:
            noisename = np.random.choice(noisefiles)
            raw_audio = read_audio(filename, noisename)
        else:
            raw_audio = read_audio(filename, noisename=None)
        feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
        if feature.ndim != 3 or feature.shape[0] < c.NUM_FRAMES or feature.shape[1] !=64 or feature.shape[2] != 1:
            print('there is an error in file:',filename)
            continue
        np.save(target_filename, feature)
        if i % 100 == 0:
            print("task:{0} cost time per audio: {1:.3f}s No.:{2} File name:{3}".format(name, time() - orig_time, i, filename))
    print("task %s runs %d seconds. %d files" %(name, time()-start_time,i))


def preprocess_and_save(wav_dir=c.WAV_DIR,out_dir=c.DATASET_DIR):

    orig_time = time()
    libri = data_catalog(wav_dir, pattern='**/*.wav')  #'/Users/walle/PycharmProjects/Speech/coding/deep-speaker-master/audio/LibriSpeechSamples/train-clean-100/19'
    if c.ADD_NOISE:
        noisefiles = find_files(c.NOISE_DIR)
    else:
        noisefiles = None
    print("extract fbank from audio and save as npy, using multiprocessing pool........ ")
    p = Pool(5)
    patch = int(len(libri)/5)
    for i in range(5):
        if i < 4:
            slibri=libri[i*patch: (i+1)*patch]
        else:
            slibri = libri[i*patch:]
        print("task %s slibri length: %d" %(i, len(slibri)))
        p.apply_async(prep, args=(slibri,out_dir,i, noisefiles))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()

    print("Extract audio features and save it as npy file, cost {0} seconds".format(time()-orig_time))
    print("*^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*  *^ˍ^*")


def test():
    libri = data_catalog()
    filename = 'audio/LibriSpeechSamples/train-clean-100/19/227/19-227-0036.wav'
    raw_audio = read_audio(filename)
    print(filename)
    feature = extract_features(raw_audio, target_sample_rate=SAMPLE_RATE)
    print(filename)


if __name__ == '__main__':
    #test()
    cvt_process_and_save("audio/test-clean/LibriSpeech/test-clean/",
                         "audio/test-clean/LibriSpeech/test-clean/")

    preprocess_and_save("audio/test-clean/LibriSpeech/test-clean/",
                         "audio/test-clean/LibriSpeech/test-clean-npy/")

