import cv2
import os
from PIL import Image
import numpy as np

import image_ops
from matplotlib import pyplot as plt

def write_video(file_path, frames, fps, dim):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h = dim[0], dim[1]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(image_ops.pil_to_cv(frame))

    writer.release()

if __name__ == '__main__':
    dir = "C:\\Data\\UnityRecorder\\Video"
    filename = "movie_008.mp4"
    save_file = "no_heuristics.mp4"
    path = dir + "\\"
    input_type = "rgb"
    dim = (1920, 1080)

    frame_num = 0
    stream = cv2.VideoCapture(path + filename)
    fps = stream.get(cv2.CAP_PROP_FPS)
    preprocessed_frames = list()
    while True:
        frame_num += 1
        ret, frame = stream.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        tmp = np.zeros(shape=dim)
        #resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # Our operations on the frame come here
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img, data = image_ops.filter_image_contours(0, frame, input_type=input_type, heuristics=False)

        # Display the resulting frame
        print("Frame {0}".format(frame_num))
        if img is not None:
            tmp = img
            cv2.imshow('Img', img)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break

        new_frame = Image.fromarray(tmp).convert('RGB')
        preprocessed_frames.append(new_frame)

    stream.release()

    #preprocessed_frames = np.array(preprocessed_frames)
    write_video(file_path=path+save_file, frames=preprocessed_frames, fps=fps, dim=dim)
    cv2.destroyAllWindows()
    print()

