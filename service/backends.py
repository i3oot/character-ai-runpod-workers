"""One initialized upstream model per warm worker; never execute user code."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


class Kimodo:
    def __init__(self):
        from huggingface_hub import snapshot_download
        locks = json.loads(Path('/app/source-lock.json').read_text())
        lock = locks['kimodo_model']
        checkpoints = Path(os.environ['MODEL_CACHE']) / 'checkpoints'
        snapshot_download(repo_id=lock['repository'], revision=lock['revision'],
            local_dir=str(checkpoints / 'Kimodo-SOMA-RP-v1.2'))
        os.environ['CHECKPOINT_DIR'] = str(checkpoints)
        encoder_lock = locks['text_encoder']
        encoder_path = snapshot_download(repo_id=encoder_lock['repository'], revision=encoder_lock['revision'],
            local_dir=str(Path(os.environ['MODEL_CACHE']) / 'matbee-llm2vec-nf4'))
        from text_encoder import MatbeeEncoder
        encoder = MatbeeEncoder(encoder_path)
        from kimodo import load_model
        self.model = load_model('Kimodo-SOMA-RP-v1.2', device='cuda', text_encoder=encoder)

    def generate(self, data, folder):
        import torch
        from kimodo.constraints import load_constraints_lst
        from kimodo.exports.motion_io import save_kimodo_npz
        from kimodo.exports.bvh import save_motion_bvh
        from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots
        from kimodo.tools import seed_everything
        seed_everything(data.get('seed', 42))
        constraints = []
        if data.get('constraints'):
            path = folder / 'constraints.json'
            path.write_text(json.dumps(data['constraints']))
            constraints = load_constraints_lst(str(path), self.model.skeleton)
        output = self.model([data['prompt']], [int(data.get('duration', 3) * self.model.fps)],
            constraint_lst=constraints, num_denoising_steps=data.get('steps', 100),
            num_samples=1, multi_prompt=True, post_processing=True, return_numpy=True)
        single = {k: v[0] if hasattr(v, 'shape') and len(v.shape) and v.shape[0] == 1 else v
                  for k, v in output.items()}
        save_kimodo_npz(str(folder / 'motion.npz'), single)
        skeleton = self.model.skeleton
        if isinstance(skeleton, SOMASkeleton30):
            skeleton = skeleton.somaskel77.to('cuda')
        positions = torch.from_numpy(output['posed_joints'][0]).to('cuda')
        rotations = torch.from_numpy(output['global_rot_mats'][0]).to('cuda')
        save_motion_bvh(str(folder / 'motion.bvh'), global_rots_to_local_rots(rotations, skeleton),
            positions[:, skeleton.root_idx, :], skeleton=skeleton, fps=self.model.fps,
            standard_tpose=True)
        return [folder / 'motion.npz', folder / 'motion.bvh']


class SkinTokens:
    def __init__(self):
        os.chdir('/opt/skintokens')
        cache = Path(os.environ['MODEL_CACHE']) / 'skintokens'
        cache.mkdir(parents=True, exist_ok=True)
        for name in ('experiments', 'models'):
            (cache / name).mkdir(exist_ok=True)
            target = Path(name)
            if not target.exists():
                target.symlink_to(cache / name, target_is_directory=True)
        subprocess.run([sys.executable, 'download.py', '--model'], check=True)
        sys.path.insert(0, '/opt/skintokens')
        import demo
        self.demo = demo
        self.server = demo.start_bpy_server()
        deadline = time.monotonic() + 180
        while True:
            if self.server.poll() is not None:
                raise RuntimeError('Local Blender helper failed')
            try:
                with socket.create_connection(('127.0.0.1', 59876), timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('Local Blender helper did not become ready') from None
                time.sleep(1)
        demo.load_model(demo.MODEL_CKPTS[0], None)

    def generate(self, data, folder):
        import torch
        from lightning import seed_everything
        seed_everything(data.get('seed', 42), workers=True)
        output = folder / 'rigged.glb'
        with torch.inference_mode():
            self.demo.run_rig([folder / 'mesh.glb'], 5, 0.95, 1.0, 1.0,
                data.get('num_beams', 5), data.get('use_skeleton', False), True, False,
                [output], self.demo.MODEL_CKPTS[0], None)
        if not output.is_file():
            raise RuntimeError('Rig export did not produce a file')
        return [output]
