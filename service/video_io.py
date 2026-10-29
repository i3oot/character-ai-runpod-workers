"""Bounded local video validation and normalization; no remote URLs or audio."""
import json
import math
from pathlib import Path
import subprocess


def inspect_video(path):
    path = Path(path)
    if path.suffix.lower() not in ('.mp4', '.mov', '.webm', '.mkv'):
        raise ValueError('Video must be MP4, MOV, WebM or MKV')
    if not path.is_file() or not 20 <= path.stat().st_size <= 256 * 1024 * 1024:
        raise ValueError('Video must be 20 bytes–256 MiB')
    result = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe',
        '-select_streams', 'v:0', '-show_entries', 'stream=width,height,duration:format=duration',
        '-of', 'json', str(path.resolve())], check=True, capture_output=True, text=True, timeout=30)
    doc = json.loads(result.stdout)
    if not doc.get('streams'):
        raise ValueError('Video has no video stream')
    stream = doc['streams'][0]
    width, height = int(stream['width']), int(stream['height'])
    duration = float(stream.get('duration', doc.get('format', {}).get('duration', 'nan')))
    if not 16 <= width <= 8192 or not 16 <= height <= 8192 or width * height > 3840 * 2160:
        raise ValueError('Input video exceeds 4K pixel budget')
    if not math.isfinite(duration) or not 0 < duration <= 86400:
        raise ValueError('Video has no valid duration, or exceeds 24 hours')
    return {'width': width, 'height': height, 'duration': duration, 'bytes': path.stat().st_size}


def validate_interval(info, start, duration):
    if (type(start) not in (int, float) or type(duration) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(duration)
            or start < 0 or not 0.5 <= duration <= 10):
        raise ValueError('Choose a finite start >= 0 and 0.5–10 seconds of video')
    if start + duration > info['duration'] + 0.001:
        raise ValueError('Requested interval extends beyond the video')


def normalize_clip(source, target, data):
    info = inspect_video(source)
    start, duration = data.get('start', 0), data.get('duration', 3)
    validate_interval(info, start, duration)
    side = data.get('max_side', 720)
    # Preserve aspect ratio, cap the longest side, never enlarge a small input.
    scale = min(1, side / max(info['width'], info['height']))
    width, height = [max(2, int(value * scale) // 2 * 2) for value in (info['width'], info['height'])]
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
        '-protocol_whitelist', 'file,pipe', '-ss', str(start), '-i', str(Path(source).resolve()),
        '-t', str(duration), '-map', '0:v:0', '-vf', f'scale={width}:{height},fps=30',
        '-frames:v', str(round(duration * 30)), '-an', '-sn', '-dn', '-map_metadata', '-1',
        '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(target)], check=True, capture_output=True, timeout=120)
    return {'source': info, 'start_seconds': start, 'requested_duration_seconds': duration,
            'normalized': inspect_video(target)}
