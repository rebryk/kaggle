import argparse
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

COLORS = ['red', 'green', 'blue', 'yellow']


def get_config():
    parser = argparse.ArgumentParser('Convert external data for Human Protein Atlas Image Classification Challenge')
    parser.add_argument('--csv', type=str, required=True, help='path to csv file')
    parser.add_argument('--size', type=int, required=True, help='new image size')
    parser.add_argument('--src', type=str, required=True, help='path to folder with images')
    parser.add_argument('--dest', type=str, default=None, help='path to folder with converted images')
    parser.add_argument('--n_threads', type=int, default=4, help='number of threads')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    return parser.parse_args()


def get_batches(values, batch_size):
    for start in range(0, len(values), batch_size):
        yield values[start: start + batch_size]


def convert_sample(args):
    name, size, src, dest = args

    for color in COLORS:
        src_path = Path(src) / f'{name}_{color}.jpg'
        dest_path = Path(dest) / f'{name}_{color}.png'
        img = Image.open(src_path)
        img = img.convert('RGB')
        img.thumbnail((size, size), Image.ANTIALIAS)
        img.save(dest_path)


if __name__ == '__main__':
    config = get_config()

    df = pd.read_csv(config.csv)
    print(f'#images: {len(df)}')

    tasks = [(it, config.size, config.src, config.dest or config.src) for it in df.Id.values]
    batches = list(get_batches(tasks, batch_size=config.batch_size))

    pool = Pool(config.n_threads)

    for batch in tqdm(batches):
        pool.map(convert_sample, batch);
