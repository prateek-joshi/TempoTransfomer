from pydub import AudioSegment
from glob import glob
import pandas as pd
import argparse
import tqdm
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_path',required=True,help='Path to folder containing all wav files')
parser.add_argument('-t','--time_split',required=False,default=10,help='Length of each split in seconds')
parser.add_argument('-r','--remove', action=argparse.BooleanOptionalAction, help='Set flag to delete original files')
args = parser.parse_args()

def split_audio(audio_path, seconds_per_split):
    start_time = 0
    end_time = (seconds_per_split * 1000) + 1
    audio = None
    try:
        audio = AudioSegment.from_wav(audio_path)
    except:
        print(f'Could not find file {audio_path}')
        return
    counter = 0
    while start_time<len(audio):
        # end_time = start_time + (seconds_per_split*1000)
        # optional check to not include short ends of audio
        newaudio = audio[start_time:end_time]
        if len(newaudio) < (end_time-start_time):
            # print('Skipped')
            break
        counter += 1
        start_time = end_time + 1
        end_time = start_time + (seconds_per_split*1000)
        new_filename = audio_path[:-4]+'-'+str(counter)+'.wav'
        newaudio.export(new_filename, format="wav")
        # print('Done')

if __name__=='__main__':
    audio_paths = glob(os.path.join(args.data_path,'ExtendedBallroom','songs','**','*.wav'), recursive=True)
    for path in tqdm.tqdm(audio_paths):
        split_audio(path, int(args.time_split))
        if args.remove:
            os.remove(path)
    print('Done')