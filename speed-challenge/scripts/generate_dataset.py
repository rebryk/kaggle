import argparse
import os

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Split video to set of images')
    parser.add_argument('-i', '--input', type=str, required=True, help='path to input video file')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to output folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    video = cv2.VideoCapture(args.input)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    for index in tqdm(range(frame_count), desc='Image processing'):
        _, image = video.read()
        cv2.imwrite(f'{args.output}/{index}.jpg', image)
