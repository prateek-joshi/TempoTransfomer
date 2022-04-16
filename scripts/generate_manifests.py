import xml.etree.ElementTree as ET
import pandas as pd
import argparse
import librosa
import logging
import math
import sys
import os
logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('generate_manifests')

parser = argparse.ArgumentParser()
# parser.add_argument('-i',required=True,help='Path to folder containing all audio files')
parser.add_argument('-o',required=False,default=os.path.join('E:','TempoTransformer','data','manifest'))
args = parser.parse_args()

def open_xml(xml_path):
    tree = ET.parse(xml_path).getroot()
    return tree

def duration(audiopath):
    y, sr = librosa.load(audiopath)
    return math.floor(librosa.get_duration(y=y, sr=sr))

def get_details_gtzan(gtzan_stats_path):
    gtzan_df = pd.read_csv(gtzan_stats_path,usecols=['filename','tempo mean'])
    gtzan_df.columns = ['audio_filepath', 'tempo']
    gtzan_df['genre'] = gtzan_df.loc[:,'audio_filepath'].str.split('.').str[0]
    gtzan_df['audio_filepath'] = os.path.join('E:\\','TempoTransformer','data','GTZAN','songs')+'\\'+gtzan_df['genre']+'\\'+gtzan_df['audio_filepath']
    gtzan_df['duration'] = gtzan_df.loc[:,'audio_filepath'].apply(duration)
    logging.info(f'Found {len(gtzan_df)} audio files in {gtzan_stats_path}')
    # logging.info('Written GTZAN manifest')
    return gtzan_df

def get_details_ebroom(root):
    songs_folder = os.path.join('E:\\','TempoTransformer','data','ExtendedBallroom','songs')
    ebroom_df = pd.DataFrame(columns=['audio_filepath','tempo','genre','duration'])
    for genre_node in root:
        genre_folder = os.path.join(songs_folder, genre_node.tag)
        for song_node in genre_node:
            song_path = os.path.join(genre_folder, song_node.get('id') + '.wav')
            if not os.path.exists(song_path):
                logging.exception(f'Could not find file {song_path}')
                continue
            try:
                song_duration = duration(song_path)
                bpm = int(song_node.get('bpm'))
                genre = genre_node.tag
                ebroom_df.loc[len(ebroom_df.index)] = [song_path, bpm, genre, song_duration]
            except:
                logging.exception(f'Could not extract details of {song_path}')
        logging.info(f'Finished {genre_node.tag}')
    logging.info(f'Found {len(ebroom_df)} audio files in {songs_folder}')
    return ebroom_df
    # logging.info('Written ExtendedBallroom manifest')

    
    # tempo = int(song_node.get('bpm'))
    # filename = os.path.join(genre_folder,song_node.get('id')+'.wav')
    # if not os.path.exists(filename):
    #     return
    # audio_duration = duration(filename)
    # datalist.append([filename, tempo, genre_node.tag, audio_duration])
    # ebroom_df = pd.DataFrame(datalist, columns=['audio_filepath','tempo','genre','duration'])
    # logging.info(f'Found {len(ebroom_df)} audio files in {ebroom_stats_path}')
    # return ebroom_df



if __name__=='__main__':
    # ANNOT_PATH = {
    #     'gtzan': 'E:\\TempoTransformer\\data\\GTZAN\\GTZAN-Rhythm_v2_ismir2015_lbd\\stats.csv',
    #     'ebroom': 'E:\\TempoTransformer\\data\\ExtendedBallroom\\songs\\extendedballroom_v1.1.xml'
    # }
    # logging.info('Starting...')
    # gtzan_df = get_details_gtzan(ANNOT_PATH['gtzan'])
    # logging.info('Retrieved GTZAN data')

    # root = open_xml(ANNOT_PATH['ebroom'])
    # ebroom_df = get_details_ebroom(root)
    # logging.info('Retrieved ExtendedBallroom data')
    # final_df = pd.concat([gtzan_df, ebroom_df])
    # print(final_df.head())
    # print('\n')
    # print(final_df.tail())
    # final_df.to_pickle(args.o, compression=None)
    # logging.info(f'Written {len(final_df)} records to {args.o}')
    df = pd.read_pickle('E:\\TempoTransformer\\data\\final_manifest.pkl', compression=None)
    logging.info(f'Read {len(df.index)} records from input manifest')
    new_df = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        counter = 1
        while counter<3:
            new_filepath = row['audio_filepath'][:-4] + '-' + str(counter) + '.wav'
            tempo = row['tempo']
            genre = row['genre']
            new_duration = 15
            if os.path.exists(new_filepath):
                new_df.loc[len(new_df.index)] = [new_filepath, tempo, genre, duration]
            counter += 1
                    

    logging.info(f'New number of records: {len(new_df.index)}')
    print(new_df.tail()['audio_filepath'].to_list())
    print()
    print(df.tail()['audio_filepath'].to_list())
    new_df.to_csv(args.o, index=False)