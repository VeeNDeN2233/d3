#!/usr/bin/env python3
"""Утилита для перекодирования существующих видео в H.264"""
import subprocess
from pathlib import Path
import sys

def reencode_video(video_path: Path):
    """Перекодирует видео в H.264"""
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return False
    
    h264_output = video_path.parent / f"{video_path.stem}_h264.mp4"
    
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
        '-movflags', '+faststart', '-pix_fmt', 'yuv420p',
        str(h264_output)
    ]
    
    print(f"Reencoding {video_path} to H.264...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0 and h264_output.exists() and h264_output.stat().st_size > 0:
        video_path.unlink()
        h264_output.rename(video_path)
        new_size = video_path.stat().st_size
        print(f"Success! Video reencoded to H.264 ({new_size / 1024 / 1024:.2f} MB)")
        return True
    else:
        print(f"Error reencoding: {result.stderr[:500] if result.stderr else 'unknown error'}")
        if h264_output.exists():
            h264_output.unlink()
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
        reencode_video(video_path)
    else:
        # Перекодируем все видео в results
        results_dir = Path("results")
        for video_file in results_dir.rglob("video_with_skeleton.mp4"):
            print(f"\nProcessing: {video_file}")
            reencode_video(video_file)

