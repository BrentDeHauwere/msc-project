# %%
import pandas as pd
import cv2
import glob
import os


INPUT_RAW_SEQ_DIR = 'data/raw/sequences/'
OUTPUT_PROCESSED_SEQ_DIR = 'data/processed/sequences/'


def raw_video_sequences_to_frames():
    # find all the videos that need to be turned into frames
    vid_names = [os.path.basename(path)
                 for path in glob.glob(f'{INPUT_RAW_SEQ_DIR}/*.dv')]
    print(f'Found {len(vid_names)} files: {vid_names}')

    # often the sequence videos have leading and trailing frames
    # these are not included in the silhouettes and should removed
    # this list identifies the start and end sequence frame
    seq_i = pd.read_csv(INPUT_RAW_SEQ_DIR + 'silhouette_frames.csv',
                        delimiter=';', index_col=0)

    # for each video, extract frames
    for vid_name in vid_names:
        # output directory per video
        vid_id = os.path.splitext(vid_name)[0]
        vid_path = OUTPUT_PROCESSED_SEQ_DIR + vid_id

        # create output directory if doesn't exist
        if not os.path.exists(vid_path):
            os.mkdir(vid_path)

        # video feed is read as VideoCapture object
        cap = cv2.VideoCapture(INPUT_RAW_SEQ_DIR + vid_name)

        # check if (camera) opened successfully
        if not cap.isOpened():
            print('ERROR: could not open video stream or file.')

        # first frame of the entire video sequence
        ret, frame = cap.read()

        # check if a starting and ending frame index was specified
        if vid_id not in seq_i.index:
            print(f'WARNING: no entry in silhouette_frames.csv for {vid_id}')

        # starting frame index, 0 if not specified
        start_i = seq_i.loc[vid_id, 'start_i'] if vid_id in seq_i.index else 0

        # last frame index, all if not specified
        end_i = seq_i.loc[vid_id, 'end_i'] if vid_id in seq_i.index else cap.get(
            cv2.CAP_PROP_FRAME_COUNT)

        # current sequence frame index
        i = 0

        # iterate over each frame
        # ret will be false when the frame is not read correctly (e.g. end of the video)
        while ret:
            # save frame as PNG file, if corresponding silhouette frame exists
            if start_i <= i < end_i:
                cv2.imwrite(f'{vid_path}/{i-start_i:06d}.png', frame)

            # retrieve next frame in video sequence
            ret, frame = cap.read()

            # increment frame index
            i += 1


raw_video_sequences_to_frames()
