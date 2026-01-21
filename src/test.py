import argparse
import cv2
import numpy as np

import os

def read_frame_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def fit_frame_to_tile(frame, tile_size, color=(0, 0, 0)):
    h, w = frame.shape[:2]
    scale = min(tile_size / w, tile_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    # pad to square tile_size
    top = (tile_size - nh) // 2
    bottom = tile_size - nh - top
    left = (tile_size - nw) // 2
    right = tile_size - nw - left
    tiled = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return tiled

def make_grid(frames, rows=6, cols=6, tile_size=160, border=20, gap=6, bg_color=(255, 255, 255)):
    h = border * 2 + rows * tile_size + (rows - 1) * gap
    w = border * 2 + cols * tile_size + (cols - 1) * gap
    grid = np.full((h, w, 3), bg_color, dtype=np.uint8)

    i = 0
    for r in range(rows):
        for c in range(cols):
            y = border + r * (tile_size + gap)
            x = border + c * (tile_size + gap)
            if i < len(frames) and frames[i] is not None:
                grid[y:y + tile_size, x:x + tile_size] = frames[i]
            else:
                # leave as background (边框留白)
                pass
            # draw tile border to make tile size visible
            cv2.rectangle(grid, (x, y), (x + tile_size - 1, y + tile_size - 1), (0, 0, 0), 1)
            i += 1
    return grid

def sample_frames_from_video(video_path, every=5, needed=36, start_frame=0):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = start_frame
    while len(frames) < needed and idx < total:
        f = read_frame_at(cap, idx)
        if f is None:
            break
        frames.append(f)
        idx += every
    cap.release()
    return frames


def parse_args():
    p = argparse.ArgumentParser(description='从视频中每隔 N 帧取一帧，并显示 6x6 网格。')
    p.add_argument('--video', type=str, default="data/truncated_videos/0000.mp4", help='视频文件路径')
    p.add_argument('--every', type=int, default=30, help='每隔多少帧取一帧 (默认 5)')
    p.add_argument('--rows', type=int, default=6)
    p.add_argument('--cols', type=int, default=6)
    p.add_argument('--tile', type=int, default=160, help='每格像素大小 (正方形)')
    p.add_argument('--border', type=int, default=20, help='外边距像素')
    p.add_argument('--gap', type=int, default=6, help='格子间间隙像素')
    return p.parse_args()

def main():
    args = parse_args()
    needed = args.rows * args.cols
    raw_frames = sample_frames_from_video(args.video, every=args.every, needed=needed, start_frame=0)

    # fit frames to tiles and fill up to needed
    tiles = []
    for f in raw_frames:
        tiles.append(fit_frame_to_tile(f, args.tile))
    # pad remaining with None so background shows (边框留白)
    while len(tiles) < needed:
        tiles.append(None)
    grid = make_grid(tiles, rows=args.rows, cols=args.cols, tile_size=args.tile, border=args.border, gap=args.gap, bg_color=(255, 255, 255))

    win_name = f'Grid {args.rows}x{args.cols} - {os.path.basename(args.video)}'
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # ensure window matches image size so tile size change is visible
    h, w = grid.shape[:2]
    try:
        cv2.resizeWindow(win_name, w, h)
    except Exception:
        pass
    cv2.imshow(win_name, grid)
    print(f'tile={args.tile}, grid={w}x{h}')
    print('按 q 或 ESC 关闭窗口')
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
