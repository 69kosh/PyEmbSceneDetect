import ffmpeg
import cv2
import numpy as np
import os
from pytube import YouTube


def prepareGeneratorFromVideo(filename: str, shape: tuple[int, int] = (160, 160), every: int = 1):

    process = (
        ffmpeg
        .input(filename)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(shape[0], shape[1]))
        .run_async(pipe_stdout=True)
    )

    i = 0
    while True:
        in_bytes = process.stdout.read(shape[0] * shape[1] * 3)
        if not in_bytes:
            break

        i += 1
        if ((i - 1) % every) != 0:
            continue

        image = np.frombuffer(in_bytes, np.uint8).reshape(
            [shape[0], shape[1], 3])
        # cv2.imshow('image', image)
        # cv2.waitKey(1)

        yield image


def generateDemoVideo(inputFilename: str, outputFilename: str, scenes: list, every = 1):

    frames = []

    for s, scene in enumerate(scenes):
        for i in range(scene[0] * every, (scene[1] + 1) * every):
            frames.append((s, scene))

    probe = ffmpeg.probe(inputFilename)
    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    width = int(video_stream['width']) #// 2
    height = int(video_stream['height']) #// 2

    # print(width, height)
    # return
    process1 = (
        ffmpeg
        .input(inputFilename)
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
        .run_async(pipe_stdout=True)
    )

    process2 = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s='{}x{}'.format(width, height))
        .output(outputFilename, pix_fmt='yuv420p')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    i = 0
    while True:
        in_bytes = process1.stdout.read(width * height * 3)
        if not in_bytes:
            break

        in_frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )

        s = frames[i][0]
        e = frames[i][1][1]

        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 40)
        fontScale = 1
        fontColor = (255, 255, 255)
        thickness = 2
        lineType = 2

        cv2.putText(in_frame, 'scene {s:03d}: {i:06d} -> {n:06d}'.format(s=s + 1, i=i + 1, n=e * every + 1),
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

        out_frame = in_frame 

        i += 1

        process2.stdin.write(
            out_frame
            .astype(np.uint8)
            .tobytes()
        )

    process2.stdin.close()
    process1.wait()
    process2.wait()


def downloadVideoIfNeed(url: str, filename: str):
    if os.path.exists(filename):
        return

    YouTube(url).streams.filter(progressive=True, file_extension='mp4').order_by(
        'resolution').desc().first().download(filename=filename)
