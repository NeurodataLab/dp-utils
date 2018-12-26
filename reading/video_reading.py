import subprocess
import ffmpeg
import numpy as np
import re
import logging
from decimal import Decimal

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


def get_video_length(path):
    process = subprocess.Popen(['/usr/bin/ffprobe', '-i', path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    matches = re.search(
        r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout, re.DOTALL).groupdict()

    hours = Decimal(matches['hours'])
    minutes = Decimal(matches['minutes'])
    seconds = Decimal(matches['seconds'])

    total = 0
    total += 60 * 60 * hours
    total += 60 * minutes
    total += seconds
    return total


def get_fps(path):
    process = subprocess.Popen(['/usr/bin/ffprobe', '-i', path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    matches = re.search(
        r", (?P<fps>\d+(\.\d+)?)\s+fps", stdout, re.DOTALL).groupdict()
    fps = Decimal(matches['fps'])
    return int(round(fps))


def frame_array_from_video(video_path, ts_start=0., ts_end=None, drop_frames_fps=None):
    """
    :param video_path: path to video
    :param ts_start: start timestamp in seconds
    :param ts_end: end timestamp in seconds
    :param drop_frames_fps: if not None, then will drop or add frames to make video 25fps
    :return: array of shape (t, h, w, 3)
    """
    logger.debug('read {}'.format(video_path))
    ts_end = ts_end or get_video_length(video_path)

    probe = ffmpeg.probe(video_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    w = int(video_stream['width'])
    h = int(video_stream['height'])

    stream = ffmpeg.input(video_path)

    stream = ffmpeg.trim(stream=stream, start=ts_start, end=ts_end)
    out, _ = stream.output('pipe:', format='rawvideo', pix_fmt='rgb24').run(capture_stdout=True)
    video = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])

    if drop_frames_fps is not None:
        fps = get_fps(video_path)
        fps_ratio = float(fps) / float(drop_frames_fps)

        num_frames_to_take = video.shape[0] / fps_ratio
        frames_getter = np.linspace(0, video.shape[0], num=int(num_frames_to_take), endpoint=False).astype(int)

        video = video[frames_getter, :]

    logger.debug('read end {}'.format(video_path))

    return video
