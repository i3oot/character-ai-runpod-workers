#!/usr/bin/env python3
"""Idempotent RunPod v2 setup; runtime secrets never enter deployment manifests."""
import argparse
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

from client import DEFAULT_DEPLOYMENT, DEFAULT_ENV, load_env, save

SERVICE_NAMES = {'kimodo': 'party-kimodo-motion', 'skintokens': 'party-skintokens-rigging',
                 'gemx': 'party-gemx-video-motion'}
CACHE_GB = {'kimodo': 25, 'skintokens': 10, 'gemx': 10}
GPU_POOLS = {'kimodo': 'AMPERE_16', 'skintokens': 'AMPERE_24', 'gemx': 'AMPERE_16'}
REGION = 'EU-RO-1'


def request(path, payload=None, method=None):
    req = urllib.request.Request('https://api.runpod.io/v2/' + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'],
                 'Content-Type': 'application/json', 'User-Agent': 'party-character-ai/1'}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # API responses may include environment values. Never print bodies or request objects.
        raise RuntimeError(f'RunPod management HTTP {error.code} for {path.split("/")[0]}') from None


def env_for(kind):
    env = {key: os.environ[key] for key in ('S3_BUCKET', 'S3_ENDPOINT_URL', 'S3_REGION',
        'S3_ACCESS_KEY_ID', 'S3_SECRET_ACCESS_KEY', 'S3_ADDRESSING_STYLE', 'S3_SESSION_TOKEN')
        if os.environ.get(key)}
    env.update(WORKER_KIND=kind, MODEL_CACHE='/runpod-volume/cache',
        HF_HOME='/runpod-volume/cache/huggingface', TORCH_HOME='/runpod-volume/cache/torch',
        XDG_CACHE_HOME='/runpod-volume/cache/xdg', HF_HUB_DISABLE_TELEMETRY='1')
    if kind == 'kimodo':
        env.update(TEXT_ENCODER_MODE='local', TEXT_ENCODER_DEVICE='cuda')
        # The pinned Matbee export and Kimodo checkpoint are public; no HF token.
    return env


def endpoint_spec(kind, volume, image_tag):
    return {'name': SERVICE_NAMES[kind], 'type': 'QUEUE',
        'image': f'ghcr.io/i3oot/{kind}-runpod-worker:{image_tag}', 'disk': 30,
        'ports': [], 'env': env_for(kind),
        'gpu': {'pools': [GPU_POOLS[kind]],
            'count': 1, 'minCudaVersion': '12.8',
            'excludedTypes': ['NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb'] if kind == 'skintokens' else []},
        'workers': {'min': 0, 'max': 1, 'idleTimeout': 5},
        'scaling': {'type': 'QUEUE_DELAY', 'queueDelay': 4},
        'timeout': 1200000, 'flashboot': 'OFF', 'dataCenterIds': [REGION],
        'mounts': {'network': [{'volumeId': volume}]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'volumes', 'apply', 'status'])
    p.add_argument('--env-file', type=Path, default=DEFAULT_ENV)
    p.add_argument('--state', type=Path, default=DEFAULT_DEPLOYMENT)
    p.add_argument('--image-tag', default='latest')
    p.add_argument('--service', choices=list(SERVICE_NAMES), action='append')
    args = p.parse_args()
    if args.command == 'apply' and not re.fullmatch('[a-f0-9]{40}', args.image_tag):
        raise ValueError('Deploy an immutable, verified 40-character image commit tag')
    if args.command == 'plan':
        selected = args.service or list(SERVICE_NAMES)
        print(json.dumps({'data_center': REGION, 'gpu_pools': {k: GPU_POOLS[k] for k in selected}, 'max_workers_each': 1,
            'min_workers_each': 0, 'idle_seconds': 5, 'model_cache_gb': {k: CACHE_GB[k] for k in selected},
            'storage_estimate_usd_month': round(sum(CACHE_GB[k] for k in selected) * 0.07, 2),
            'gpu_estimate_usd_running_hour': {k: (0.684 if GPU_POOLS[k] == 'AMPERE_24' else 0.576) for k in selected}}, indent=2))
        return
    load_env(args.env_file)
    endpoints = request('serverless')['endpoints']
    volumes = request('network-volumes')['networkVolumes']
    state = json.loads(args.state.read_text()) if args.state.exists() else {'services': {}}
    for kind, name in SERVICE_NAMES.items():
        if args.service and kind not in args.service:
            continue
        cache_name = f'{name}-cache-{REGION.lower()}'
        matches = [v for v in volumes if v['name'] == cache_name]
        existing = [e for e in endpoints if e['name'] == name]
        if len(matches) > 1 or len(existing) > 1:
            raise RuntimeError('Duplicate service/volume names; inspect before continuing')
        volume = matches[0] if matches else None
        endpoint = existing[0] if existing else None
        if args.command in ('volumes', 'apply') and volume is None:
            volume = request('network-volumes', {'name': cache_name, 'size': CACHE_GB[kind],
                'dataCenter': REGION, 'type': 'STANDARD'})
            state['services'][kind] = {'volume_id': volume['id'], 'cache_gb': volume['size']}
            save(args.state, state)
        if args.command == 'apply':
            if not re.fullmatch('[a-f0-9]{40}', args.image_tag):
                raise ValueError('Deploy an immutable, verified 40-character image commit tag')
            if endpoint is None:
                endpoint = request('serverless', endpoint_spec(kind, volume['id'], args.image_tag))
            else:
                # Existing resources are inspected, never silently reset or rotated on rerun.
                print(f'{kind}: existing endpoint retained')
        if endpoint:
            details = request('serverless/' + endpoint['id'])
            state['services'][kind] = {'endpoint_id': endpoint['id'],
                'url': f'https://console.runpod.io/serverless/user/endpoint/{endpoint["id"]}',
                'volume_id': volume['id'] if volume else None,
                'cache_gb': volume['size'] if volume else None,
                'workers': details.get('workers'), 'gpu': details.get('gpu'),
                'image': details.get('image'), 'data_centers': details.get('dataCenterIds'),
                'hf_token_configured': bool(details.get('env', {}).get('HF_TOKEN')) if kind == 'kimodo' else None}
        elif volume:
            state['services'][kind] = {'volume_id': volume['id'], 'cache_gb': volume['size']}
        save(args.state, state)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError) as error:
        if isinstance(error, OSError):
            raise SystemExit('Network/filesystem operation failed; no automatic provisioning retry') from None
        raise SystemExit(str(error)) from None
