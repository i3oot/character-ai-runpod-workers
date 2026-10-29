"""Private S3 artifacts, persistent model cache, serialized scale-to-zero workers."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time

from protocol import PREFIX, digest, identifier, validate
from storage import bucket_name, read_result, s3_client

LOCK = threading.Lock()
BACKEND = None


def validate_glb(path, *, rigged=False):
    import struct
    with path.open('rb') as stream:
        magic, version, size = struct.unpack('<4sII', stream.read(12))
        length, chunk = struct.unpack('<I4s', stream.read(8))
        if magic != b'glTF' or version != 2 or size != path.stat().st_size or chunk != b'JSON':
            raise ValueError('Invalid GLB container')
        if length > 64 * 1024 * 1024:
            raise ValueError('GLB JSON exceeds limit')
        doc = json.loads(stream.read(length))
    if not doc.get('meshes'):
        raise ValueError('GLB has no mesh')
    for entry in doc.get('buffers', []) + doc.get('images', []):
        uri = entry.get('uri', '')
        if uri and not uri.startswith('data:'):
            raise ValueError('External GLB resources are unsupported')
    if rigged and (not doc.get('skins') or not any(s.get('joints') for s in doc['skins'])):
        raise ValueError('Output GLB has no skeleton/skin')
    return doc


def handler(job):
    global BACKEND
    kind = os.environ['WORKER_KIND']
    data = validate(job['input'], kind)
    run = identifier(job['id'])
    root = f'{PREFIX}/outputs/{kind}/{run}'
    storage, bucket = s3_client(), bucket_name()
    with LOCK:
        existing = read_result(storage, bucket, root + '/result.json')
        if existing:
            return existing
        with tempfile.TemporaryDirectory(prefix=kind + '-', dir='/tmp') as temp:
            folder = Path(temp)
            if kind == 'gemx':
                obj = storage.head_object(Bucket=bucket, Key=data['video_key'])
                if not 20 <= obj['ContentLength'] <= 256 * 1024 * 1024:
                    raise ValueError('Input video exceeds 256 MiB or is empty')
                video = folder / ('video' + Path(data['video_key']).suffix)
                storage.download_file(bucket, data['video_key'], str(video))
                if digest(video) != data['video_sha256']:
                    raise ValueError('Input checksum mismatch')
                from video_io import inspect_video, validate_interval
                validate_interval(inspect_video(video), data.get('start', 0), data.get('duration', 3))
            if kind == 'skintokens':
                obj = storage.head_object(Bucket=bucket, Key=data['mesh_key'])
                if not 20 <= obj['ContentLength'] <= 128 * 1024 * 1024:
                    raise ValueError('Input GLB exceeds 128 MiB or is empty')
                mesh = folder / 'mesh.glb'
                storage.download_file(bucket, data['mesh_key'], str(mesh))
                if digest(mesh) != data['mesh_sha256']:
                    raise ValueError('Input checksum mismatch')
                doc = validate_glb(mesh)
                if data.get('use_skeleton') and not doc.get('skins'):
                    raise ValueError('Skin-only mode requires an existing skeleton')
            if BACKEND is None:
                from backends import Kimodo, SkinTokens
                if kind == 'gemx':
                    from gemx_backend import GemX
                    BACKEND = GemX()
                else:
                    BACKEND = (Kimodo if kind == 'kimodo' else SkinTokens)()
            import torch
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            files = BACKEND.generate(data, folder)
            if kind == 'skintokens':
                validate_glb(files[0], rigged=True)
            artifacts = []
            for path in files:
                key = root + '/' + path.name
                storage.upload_file(str(path), bucket, key)
                artifacts.append({'name': path.name, 'key': key, 'sha256': digest(path), 'bytes': path.stat().st_size})
            import torch
            result = {'kind': kind, 'job_id': run, 'artifacts': artifacts,
                'generation_seconds': round(time.monotonic() - started, 3),
                'gpu': torch.cuda.get_device_name(),
                'peak_vram_bytes': torch.cuda.max_memory_allocated(),
                'source_lock': json.loads(Path('/app/source-lock.json').read_text())}
            storage.put_object(Bucket=bucket, Key=root + '/result.json',
                Body=json.dumps(result).encode(), ContentType='application/json')
            return result


if __name__ == '__main__':
    import runpod
    cache = Path(os.environ['MODEL_CACHE'])
    if not Path('/runpod-volume').is_mount():
        raise RuntimeError('A persistent model-cache network volume is required')
    cache.mkdir(parents=True, exist_ok=True)
    runpod.serverless.start({'handler': handler})
