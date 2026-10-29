#!/usr/bin/env python3
"""Submit once, retain a ticket, retrieve private artifacts without waking GPUs."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from protocol import PREFIX, digest, identifier, validate
from storage import bucket_name, read_result, s3_client

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV = ROOT / '.tools/image-to-3d/runpod/client.env'
DEFAULT_DEPLOYMENT = ROOT / '.tools/character-ai/runpod/deployment.json'


def load_env(path):
    """Read an already provisioned private config; never shell-evaluate or print it."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.removeprefix('export ').split('=', 1)
        values = shlex.split(value)
        if len(values) != 1:
            raise ValueError('Invalid private runtime config')
        os.environ.setdefault(key.strip(), values[0])


def api(endpoint, operation, payload=None):
    request = urllib.request.Request(
        f'https://api.runpod.ai/v2/{identifier(endpoint)}/{operation}',
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'],
                 'Content-Type': 'application/json', 'User-Agent': 'party-character-ai/1'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'RunPod HTTP {error.code}; no automatic resubmission') from None


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        pending = Path(stream.name)
    pending.replace(path)


def prepare(args):
    if args.kind == 'kimodo':
        data = {'kind': 'kimodo', 'prompt': args.prompt, 'duration': args.duration,
                'steps': args.steps, 'seed': args.seed}
        if args.constraints:
            data['constraints'] = json.loads(args.constraints.read_text())
    elif args.kind == 'gemx':
        from video_io import inspect_video, validate_interval
        inspect = inspect_video(args.video)
        validate_interval(inspect, args.start, args.duration)
        data = {'kind': 'gemx', 'video_key': f'{PREFIX}/inputs/{uuid.uuid4().hex}/video{args.video.suffix.lower()}',
                'video_sha256': digest(args.video), 'seed': args.seed,
                'start': args.start, 'duration': args.duration, 'camera': args.camera,
                'max_side': args.max_side, 'postprocess': not args.no_postprocess}
    else:
        if not args.mesh.is_file() or not 20 <= args.mesh.stat().st_size <= 128 * 1024 * 1024:
            raise ValueError('Input must be a GLB smaller than 128 MiB')
        from handler import validate_glb
        doc = validate_glb(args.mesh)
        if args.use_skeleton and not doc.get('skins'):
            raise ValueError('Skin-only mode requires an existing skeleton')
        data = {'kind': 'skintokens', 'mesh_key': f'{PREFIX}/inputs/{uuid.uuid4().hex}/mesh.glb',
                'mesh_sha256': digest(args.mesh), 'seed': args.seed,
                'use_skeleton': args.use_skeleton, 'num_beams': args.num_beams}
    return validate(data, args.kind)


def fetch(ticket_path, folder, timeout, poll):
    ticket = json.loads(ticket_path.read_text())
    if ticket.get('state') == 'submitting' and not ticket.get('job_id'):
        raise RuntimeError('Ambiguous submission: inspect RunPod before submitting again; ticket retained')
    storage = s3_client()
    started = time.monotonic()
    while True:
        result = read_result(storage, ticket['bucket'], ticket['metadata_key'])
        if result:
            folder.mkdir(parents=True, exist_ok=True)
            expected_root = f'{PREFIX}/outputs/{ticket["input"]["kind"]}/{identifier(ticket["job_id"])}/'
            for artifact in result['artifacts']:
                name = artifact['name']
                if Path(name).name != name or not artifact['key'].startswith(expected_root):
                    raise ValueError('Unexpected artifact destination')
                destination = folder / name
                if destination.exists():
                    if digest(destination) != artifact['sha256']:
                        raise ValueError('Existing output differs; choose another output directory')
                    continue
                with tempfile.NamedTemporaryFile(dir=folder, delete=False) as pending:
                    tmp = Path(pending.name)
                try:
                    storage.download_file(ticket['bucket'], artifact['key'], str(tmp))
                    if digest(tmp) != artifact['sha256'] or tmp.stat().st_size != artifact['bytes']:
                        raise ValueError('Downloaded artifact checksum/size mismatch')
                    if name.endswith('.glb'):
                        from handler import validate_glb
                        validate_glb(tmp, rigged=True)
                    os.link(tmp, destination)
                finally:
                    tmp.unlink(missing_ok=True)
            save(folder / 'result.json', result)
            print(f'Verified {len(result["artifacts"])} artifacts in {folder}')
            return
        status = api(ticket['endpoint_id'], 'status/' + identifier(ticket['job_id']))
        if status['status'] in ('FAILED', 'CANCELLED', 'TIMED_OUT'):
            raise RuntimeError(f'Job {status["status"]}; inspect worker logs; ticket retained')
        if time.monotonic() - started > timeout:
            raise TimeoutError('Local wait ended; remote job retained. Resume with fetch')
        print(status['status'], flush=True)
        time.sleep(poll)


def run(args):
    if args.env_file.exists():
        load_env(args.env_file)
    if args.command == 'fetch':
        fetch(args.ticket, args.output_dir, args.timeout, args.poll)
        return
    data = prepare(args)
    ticket_path = args.ticket or args.output_dir / 'job.json'
    endpoint = args.endpoint
    if not endpoint:
        endpoint = json.loads(args.deployment.read_text())['services'][args.kind]['endpoint_id']
    if ticket_path.exists():
        ticket = json.loads(ticket_path.read_text())
        previous = {k: v for k, v in ticket['input'].items() if k not in ('mesh_key', 'video_key')}
        requested = {k: v for k, v in data.items() if k not in ('mesh_key', 'video_key')}
        if previous != requested or ticket['endpoint_id'] != endpoint:
            raise ValueError('Ticket belongs to another input/options/endpoint')
        if ticket.get('state') == 'submitting' and not ticket.get('job_id'):
            raise RuntimeError('Inspect ambiguous submission in RunPod; automatic resubmit refused')
    else:
        storage, bucket = s3_client(), bucket_name()
        if args.kind == 'skintokens':
            storage.upload_file(str(args.mesh), bucket, data['mesh_key'])
        elif args.kind == 'gemx':
            storage.upload_file(str(args.video), bucket, data['video_key'])
        # Write BEFORE the billable request. A lost response must not trigger duplicate spending.
        ticket = {'endpoint_id': identifier(endpoint), 'bucket': bucket, 'input': data, 'state': 'submitting'}
        ticket_path.parent.mkdir(parents=True, exist_ok=True)
        with ticket_path.open('x') as stream:
            json.dump(ticket, stream, indent=2)
        reply = api(endpoint, 'run', {'input': data})
        job = identifier(reply['id'])
        ticket.update(job_id=job, state='submitted', metadata_key=f'{PREFIX}/outputs/{args.kind}/{job}/result.json')
        save(ticket_path, ticket)
    print(f'Job {ticket["job_id"]}; ticket: {ticket_path}', flush=True)
    if not args.submit_only:
        fetch(ticket_path, args.output_dir, args.timeout, args.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path, default=DEFAULT_ENV)
    parser.add_argument('--deployment', type=Path, default=DEFAULT_DEPLOYMENT)
    sub = parser.add_subparsers(dest='command', required=True)
    for kind in ('kimodo', 'skintokens', 'gemx'):
        p = sub.add_parser(kind)
        p.set_defaults(kind=kind)
        p.add_argument('--endpoint')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--ticket', type=Path)
        p.add_argument('--submit-only', action='store_true')
        if kind == 'kimodo':
            p.add_argument('prompt')
            p.add_argument('--duration', type=float, default=3)
            p.add_argument('--steps', type=int, default=100)
            p.add_argument('--constraints', type=Path)
        elif kind == 'gemx':
            p.add_argument('video', type=Path)
            p.add_argument('--start', type=float, default=0)
            p.add_argument('--duration', type=float, default=3)
            p.add_argument('--camera', choices=['static', 'background'], default='static')
            p.add_argument('--max-side', type=int, default=720)
            p.add_argument('--no-postprocess', action='store_true')
        else:
            p.add_argument('mesh', type=Path)
            p.add_argument('--use-skeleton', action='store_true')
            p.add_argument('--num-beams', type=int, default=5)
    p = sub.add_parser('fetch')
    p.add_argument('ticket', type=Path)
    for p in sub.choices.values():
        p.add_argument('-o', '--output-dir', type=Path, required=True)
        p.add_argument('--timeout', type=int, default=3600)
        p.add_argument('--poll', type=int, default=10)
    args = parser.parse_args()
    try:
        run(args)
    except (subprocess.SubprocessError, ValueError, RuntimeError, TimeoutError, OSError, KeyError) as error:
        # Runtime credentials and HTTP request objects must never be included in diagnostics.
        if isinstance(error, OSError):
            parser.exit(1, 'Network/filesystem operation failed; check runtime configuration\n')
        if isinstance(error, subprocess.SubprocessError):
            parser.exit(1, 'Video inspection failed; check FFmpeg installation and input file\n')
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
