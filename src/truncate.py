import argparse
import numpy as np
import cv2
import os
import shutil
import subprocess
import ffmpeg
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def truncate(video_name, front, back, bitrate):
    base = video_name.replace(".mp4", "")
    video_path = f"data/videos/{video_name}"
    final_path = f"data/truncated_videos/{base}.mp4"

    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])

    if shutil.which("ffmpeg") is None:
        print("ffmpeg未找到，保存至：", video_path)
        return
    
    encoder = "libx264"
    print("使用编码器：", encoder)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path
    ]
    cmd += [
        "-ss", f"{front:.2f}",
        "-t", f"{duration - front - back:.2f}",
        "-vf", "scale=960:540"
    ]
    cmd += [
        "-c:v", encoder,
        "-b:v", f"{int(bitrate)}k",
        "-pix_fmt", "yuv420p",
        final_path
    ]
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("ffmpeg 失败，stderr:\n", proc.stderr.decode(errors="ignore"))
    else:
        print("保存至：", final_path)


def main(args):
    os.makedirs("data/truncated_videos", exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = []
        for f in os.listdir("data/videos"):
            if f.endswith(".mp4"):
                futures.append(executor.submit(
                    truncate,
                    f,
                    args.front,
                    args.back,
                    args.bitrate
                ))
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate a video.")
    parser.add_argument("--front", type=int, default=10.2,
                        help="Number of frames to remove from the start.")
    parser.add_argument("--back", type=int, default=14.2,
                        help="Number of frames to remove from the end.")
    parser.add_argument("--bitrate", type=int, default=1500,
                        help="Bitrate of the output video in kbps.")
    args = parser.parse_args()
    main(args)