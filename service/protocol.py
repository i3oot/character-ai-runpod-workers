"""Small, bounded job contract shared by both character workers."""
import hashlib
import math
import re

PREFIX = 'character-ai'
KINDS = ('kimodo', 'skintokens', 'gemx')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,120}', value):
        raise ValueError('Invalid job identifier')
    return value


def validate(data, kind):
    if kind not in KINDS or not isinstance(data, dict) or data.get('kind') != kind:
        raise ValueError('Job does not match this worker')
    allowed = {
        'kimodo': {'kind', 'seed', 'prompt', 'duration', 'steps', 'constraints'},
        'skintokens': {'kind', 'seed', 'mesh_key', 'mesh_sha256', 'use_skeleton', 'num_beams'},
        'gemx': {'kind', 'seed', 'video_key', 'video_sha256', 'start', 'duration',
                 'camera', 'max_side', 'postprocess'},
    }[kind]
    if set(data) - allowed:
        raise ValueError('Unknown job option')
    seed = data.get('seed', 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid seed')
    if kind == 'kimodo':
        prompt = data.get('prompt')
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 1000:
            raise ValueError('Prompt must contain 1–1000 characters')
        # This endpoint deliberately allows one short prompt, not unbounded sequences.
        if len([p for p in prompt.split('.') if p.strip()]) != 1:
            raise ValueError('Use one sentence per job')
        duration = data.get('duration', 3)
        if type(duration) not in (int, float) or not math.isfinite(duration) or not 0.5 <= duration <= 10:
            raise ValueError('Duration must be 0.5–10 seconds')
        if type(data.get('steps', 100)) is not int or not 10 <= data.get('steps', 100) <= 200:
            raise ValueError('Steps must be 10–200')
        if 'constraints' in data:
            import json
            if not isinstance(data['constraints'], list) or len(json.dumps(data['constraints'])) > 65536:
                raise ValueError('Constraints must be a JSON list smaller than 64 KiB')
    elif kind == 'skintokens':
        key = data.get('mesh_key', '')
        if not re.fullmatch(r'character-ai/inputs/[a-f0-9]{32}/mesh\.glb', key):
            raise ValueError('Mesh must use this service input prefix')
        if not re.fullmatch('[a-f0-9]{64}', data.get('mesh_sha256', '')):
            raise ValueError('Mesh checksum required')
        if type(data.get('use_skeleton', False)) is not bool:
            raise ValueError('use_skeleton must be boolean')
        if type(data.get('num_beams', 5)) is not int or not 1 <= data.get('num_beams', 5) <= 10:
            raise ValueError('num_beams must be 1–10')
    else:
        if not re.fullmatch(r'character-ai/inputs/[a-f0-9]{32}/video\.(mp4|mov|webm|mkv)', data.get('video_key', '')):
            raise ValueError('Video must use this service input prefix')
        if not re.fullmatch('[a-f0-9]{64}', data.get('video_sha256', '')):
            raise ValueError('Video checksum required')
        for key, default, low, high in [('start', 0, 0, 86400), ('duration', 3, 0.5, 10)]:
            value = data.get(key, default)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f'{key} must be finite and within {low}–{high} seconds')
        if data.get('camera', 'static') not in ('static', 'background'):
            raise ValueError('Camera must be static or background')
        if type(data.get('max_side', 720)) is not int or not 256 <= data.get('max_side', 720) <= 1280:
            raise ValueError('max_side must be 256–1280 pixels')
        if type(data.get('postprocess', True)) is not bool:
            raise ValueError('postprocess must be boolean')
    return data
