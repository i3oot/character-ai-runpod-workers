"""Pinned GEM-X single-person video inference, using only the model volume as cache."""
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import urllib.request
from zipfile import ZipFile

from protocol import digest
from video_io import normalize_clip


class GemX:
    def __init__(self):
        from huggingface_hub import snapshot_download
        from gem.inference.body_video import load_body_video_config
        self.locks = json.loads(Path('/app/source-lock.json').read_text())
        cache = Path(os.environ['MODEL_CACHE']) / 'gemx'
        self.weights, self.soma = cache / 'weights', cache / 'soma'
        for label, destination, names in [
            ('gemx_model', self.weights, ['body_s240000.ckpt', 'body_config.json', 'vitpose.pth',
             'motion_native77_e774_s292175.ckpt', 'motion_stats_raw.npz', 'body_video_native77.yaml']),
            ('gemx_soma', self.soma, ['SOMA_neutral.npz', 'SOMA_template_rig.usda',
                                    'SOMA_procedural_transforms.json']),
        ]:
            lock = self.locks[label]
            snapshot_download(repo_id=lock['repository'], revision=lock['revision'],
                allow_patterns=names, local_dir=str(destination), token=False)
        detector = self.weights / 'yolox_humanart.onnx'
        lock = self.locks['gemx_detector']
        if not detector.exists() or digest(detector) != lock['sha256']:
            archive = self.weights / 'yolox-download.zip'
            pending = detector.with_suffix('.pending')
            try:
                with urllib.request.urlopen(lock['url'], timeout=120) as source, archive.open('wb') as target:
                    shutil.copyfileobj(source, target)
                if digest(archive) != lock['archive_sha256']:
                    raise ValueError('Person detector archive checksum mismatch')
                with ZipFile(archive) as z:
                    member = next(n for n in z.namelist() if n.endswith('/end2end.onnx'))
                    with z.open(member) as source, pending.open('wb') as target:
                        shutil.copyfileobj(source, target)
                if digest(pending) != lock['sha256']:
                    raise ValueError('Person detector checksum mismatch')
                pending.replace(detector)
            finally:
                archive.unlink(missing_ok=True)
                pending.unlink(missing_ok=True)
        self.config = load_body_video_config(self.weights / 'body_video_native77.yaml')
        # Both ~3.5GB encoders run sequentially. Keep activation batches small.
        self.config['pose']['batch_size'] = 1
        self.config['observations']['batch_size'] = 1
        self.config['detector']['device'] = 'cpu'

    def generate(self, data, folder):
        import numpy as np
        import torch
        from gem.inference.body_video import run_body_video
        from gem.inference.postprocess import postprocess_motion_result
        from gem.inference.serialization import save_predictions
        from gemx_export import export_motion, make_preview
        torch.manual_seed(data.get('seed', 42))
        np.random.seed(data.get('seed', 42))
        clip = folder / 'clip.mp4'
        selection = normalize_clip(folder / ('video' + Path(data['video_key']).suffix), clip, data)
        config = copy.deepcopy(self.config)
        config['camera']['mode'] = data.get('camera', 'static')
        with torch.inference_mode():
            result = run_body_video(clip, config, device='cuda', progress=lambda s: print(s, flush=True))
            if data.get('postprocess', True):
                result = postprocess_motion_result(result, soma_assets=self.soma, device='cuda')
            save_predictions(result, folder / 'motion.pt')
            gc.collect()
            torch.cuda.empty_cache()
            geometry = export_motion(result, self.soma, folder)
        make_preview(clip, result, geometry, folder)
        report = {'format': 'party_gemx_video_v1', 'selection': selection,
            'camera': config['camera']['mode'], 'fps': 30, 'frames': len(geometry['positions']),
            'joints': len(geometry['names']), 'coordinates': 'Y_up_meters',
            'bvh_units': 'centimeters', 'bvh_channels': '6 per joint; preserves procedural translations',
            'postprocessing': result.get('postprocessing', False),
            'subject_selection': 'primary person by cumulative detected area',
            'limitations': ['Monocular depth and occlusion remain estimates',
                'No dedicated EgoHand pass; fingers are body-model estimates',
                'Retargeting, loop creation and gameplay root-motion policy are separate steps'],
            'source_lock': {k: v for k, v in self.locks.items() if k.startswith('gemx')}}
        (folder / 'motion-report.json').write_text(json.dumps(report, indent=2))
        return [folder / name for name in ('motion.pt', 'motion.npz', 'motion.bvh',
                                           'preview.mp4', 'preview.png', 'motion-report.json')]
