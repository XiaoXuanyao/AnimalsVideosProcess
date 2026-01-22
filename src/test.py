import cv2
import os
import tqdm


cnt = 0

with tqdm.tqdm() as pbar:
    for e in os.listdir("./data/truncated_videos"):
        p = os.path.join("./data/truncated_videos", e)
        if os.path.isdir(p):
            continue
        cap = cv2.VideoCapture(p)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cnt += frame_count
        pbar.update(1)

print("Total frames:", cnt)
print("Approx total seconds:", cnt / 30)