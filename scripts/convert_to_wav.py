import tqdm
import glob, os
import subprocess
import argparse
import logging, sys
import multiprocessing as mp
logging.basicConfig(
    level=logging.DEBUG, 
    format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('convert_to_wav')

parser = argparse.ArgumentParser(description='Convert all mp3 files in a folder to wav format')
parser.add_argument('--songs_dir','-i', required=True, help='Path to directory containing all mp3 files')
parser.add_argument('--remove','-r', required=False, default=True, help='Flag to delete mp3 files after conversion, default is True')
parser.add_argument('--num_cpus','-n',required=False, default=1, help='Number of processes to use in multiprocessing, default is 1')
parser.add_argument('--bin',required=True,help='Path to ffmpeg bin folder')
args = parser.parse_args()

def saveas_wav(mp3_path):
    wav_path = mp3_path[:-4] + '.wav'
    ffmpeg_args = [
        os.path.join(args.bin, 'ffmpeg'),
        '-hide_banner',
        '-loglevel',
        'error',
        '-y',
        '-i',
        mp3_path,
        wav_path
    ]
    try:
        subprocess.call(ffmpeg_args)
        if args.remove:
            os.remove(mp3_path)
    except Exception as e:
        logger.error(e)

if __name__=='__main__':
    songs_folder = args.songs_dir
    logger.debug(songs_folder)
    logger.info(f'Looking into: {songs_folder}')
    mp3_list = glob.glob(os.path.join(songs_folder,'*','*.mp3'), recursive=True)
    logger.info(f'Found {len(mp3_list)} songs in mp3 format')
    logger.info('Converting to wav...')
    p = mp.Pool(processes=int(args.num_cpus))
    max_ = len(mp3_list)
    with tqdm.tqdm(total=max_, desc='Completion') as pbar:
        for i,_ in enumerate(p.imap_unordered(saveas_wav, mp3_list)):
            pbar.update()
    p.close()
    p.join()
    logger.info('Done!')