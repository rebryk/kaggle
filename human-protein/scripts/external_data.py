import argparse
import os
import subprocess
from multiprocessing import Pool

import pandas as pd
import requests
from tqdm import tqdm

COLORS = ['red', 'green', 'blue', 'yellow']
CSV_URL = "https://storage.googleapis.com/kaggle-forum-message-attachments/432870/10816/HPAv18RBGY_wodpl.csv"
URL = 'http://v18.proteinatlas.org/images'


def get_config():
    parser = argparse.ArgumentParser('Download external data for Human Protein Atlas Image Classification Challenge')
    parser.add_argument('--path', type=str, required=True, help='download location')
    parser.add_argument('--n_threads', type=int, required=True, help='number of threads')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='batch size')
    return parser.parse_args()


def get_image_list(df):
    for image_id in df.Id:
        first, second = image_id.split('_', 1)
        for color in COLORS:
            image_name = f'{image_id}_{color}.jpg'
            image_url = f'{URL}/{first}/{second}_{color}.jpg'
            yield (image_name, image_url)


def download_image(args):
    image_path, image_url = args

    if not os.path.isfile(image_path):
        r = requests.get(image_url, allow_redirects=True)
        open(image_path, 'wb').write(r.content)


def get_batches(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start: start + batch_size]


if __name__ == '__main__':
    config = get_config()

    # Download .csv file with dataset information
    subprocess.run(['wget', '-O', f'{config.path}/train.csv', CSV_URL])

    df = pd.read_csv(f'{config.path}/train.csv')
    print(f'#images: {len(df)}')

    image_list = list([(f'{config.path}/train/{name}', url) for (name, url) in get_image_list(df)])
    batches = list(get_batches(image_list, batch_size=config.batch_size))

    # Create the directory for images
    path = f'{config.path}/train'
    if not os.path.isdir(path):
        os.mkdir(path)

    pool = Pool(config.n_threads)

    for batch in tqdm(batches):
        pool.map(download_image, batch);
