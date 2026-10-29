"""CPU tests for cost boundaries, ticket reuse, and reconstructed BVH motion."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from protocol import validate
from video_io import inspect_video, validate_interval, normalize_clip
import client
import deploy


def request():
    return {'kind': 'gemx', 'video_key': 'character-ai/inputs/' + 'a' * 32 + '/video.mp4',
            'video_sha256': 'b' * 64, 'duration': 3, 'start': 0, 'max_side': 720,
            'camera': 'static', 'postprocess': True}


class GemXContractTests(unittest.TestCase):
    def test_bounded_private_video_jobs(self):
        data = request()
        validate(data, 'gemx')
        for key, value in [('video_key', 'https://example.com/a.mp4'), ('video_key', '../a.mp4'),
            ('video_sha256', ''), ('duration', 11), ('duration', float('nan')), ('start', -1),
            ('start', True), ('max_side', 4000), ('postprocess', 'yes'), ('camera', 'auto'),
            ('command', 'arbitrary code')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate(dict(data, **{key: value}), 'gemx')
        for kind in ['kimodo', 'skintokens']:
            with self.assertRaises(ValueError):
                validate(data, kind)

    def test_clip_is_not_silently_truncated(self):
        for start, duration in [(2, 3), (0, 11), (float('nan'), 1), (0, True)]:
            with self.assertRaises(ValueError):
                validate_interval({'duration': 4}, start, duration)
        validate_interval({'duration': 4}, 1, 3)

    def test_ticket_reuses_video_without_upload_or_paid_resubmission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(env_file=root / 'absent.env', command='gemx', kind='gemx',
                ticket=None, output_dir=root, endpoint='endpoint', submit_only=True)
            old = dict(request(), video_key='character-ai/inputs/' + 'c' * 32 + '/video.mp4')
            client.save(root / 'job.json', {'endpoint_id': 'endpoint', 'input': old,
                'state': 'submitted', 'job_id': 'existing-job'})
            with patch.object(client, 'prepare', return_value=request()), \
                patch.object(client, 'api', side_effect=AssertionError('Duplicate paid job')), \
                patch.object(client, 's3_client', side_effect=AssertionError('Duplicate upload')):
                client.run(args)
            with patch.object(client, 'prepare', return_value=dict(request(), duration=4)):
                with self.assertRaisesRegex(ValueError, 'another input'):
                    client.run(args)

    def test_small_separate_endpoint_with_no_price_fallback(self):
        with patch.object(deploy, 'env_for', return_value={'WORKER_KIND': 'gemx'}):
            spec = deploy.endpoint_spec('gemx', 'own-cache', 'a' * 40)
        self.assertEqual(spec['gpu']['pools'], ['AMPERE_16'])
        self.assertEqual(spec['workers'], {'min': 0, 'max': 1, 'idleTimeout': 5})
        self.assertEqual(spec['mounts']['network'], [{'volumeId': 'own-cache'}])
        self.assertEqual(deploy.CACHE_GB['gemx'], 10)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
    def test_real_video_normalization_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / 'input.mp4', root / 'clip.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=640x360:rate=24', '-t', '2', '-c:v', 'libx264', '-threads', '2',
                str(source)], check=True, capture_output=True)
            report = normalize_clip(source, target, dict(request(), start=0.5, duration=1, max_side=320))
            self.assertEqual(report['normalized']['width'], 320)
            self.assertEqual(report['normalized']['height'], 180)
            self.assertAlmostEqual(report['normalized']['duration'], 1, places=2)
            source.with_suffix('.txt').write_text('not a video')
            with self.assertRaises(ValueError):
                inspect_video(source.with_suffix('.txt'))
            with self.assertRaises(ValueError):
                normalize_clip(source, target, dict(request(), duration=3))

    def test_bvh_round_trip_preserves_motion_and_procedural_translations(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from gemx_export import write_bvh
        parents = np.array([-1, 0, 0, 1])
        names = ['Hips', 'Spine', 'LeftLeg', 'Head']
        rest = np.array([[0, 1, 0], [0, 1.4, 0], [-0.2, 0.9, 0], [0, 1.8, 0]])
        bind = Rotation.from_euler('XYZ', [[12, 8, 0], [0, 90, 0], [0, 0, 45], [2, 3, 4]], degrees=True).as_matrix()
        delta = Rotation.from_euler('XYZ', [[0, 0, 0], [25, 40, -15], [20, 90, 0]], degrees=True).as_matrix()
        delta = np.broadcast_to(delta[:, None], (3, 4, 3, 3)).copy()
        delta[:, 1] = delta[:, 1] @ Rotation.from_euler('Z', [0, 35, -20], degrees=True).as_matrix()
        positions = np.einsum('tij,kj->tki', delta[:, 0], rest - rest[0]) + [1, 1, 0.3]
        positions[:, 3] += np.array([[0, 0, 0], [0.01, 0.02, 0], [-0.02, 0, 0.03]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'motion.bvh'
            write_bvh(path, names, parents, positions, delta @ bind, rest, bind)
            text = path.read_text()
            self.assertIn('Frames: 3', text)
            # Parse the serialized motion, in BVH's depth-first joint order.
            lines = text.split('Frame Time: ')[1].splitlines()[1:]
            packed = np.array([[float(v) for v in line.split()] for line in lines]).reshape(3, 4, 6)
            order = [0, 1, 3, 2]
            self.assertLess(text.index('JOINT Head'), text.index('JOINT LeftLeg'))
            channels = packed[:, np.argsort(order)]
            reconstructed = np.empty_like(positions)
            world_rotation = np.empty_like(delta)
            for j, parent in enumerate(parents):
                local = Rotation.from_euler('XYZ', channels[:, j, 3:], degrees=True).as_matrix()
                offset = rest[j] if parent < 0 else rest[j] - rest[parent]
                translation = channels[:, j, :3] / 100 + offset
                if parent < 0:
                    reconstructed[:, j], world_rotation[:, j] = translation, local
                else:
                    reconstructed[:, j] = reconstructed[:, parent] + np.einsum('tij,tj->ti', world_rotation[:, parent], translation)
                    world_rotation[:, j] = world_rotation[:, parent] @ local
            np.testing.assert_allclose(reconstructed, positions, atol=1e-8)
            np.testing.assert_allclose(world_rotation, delta, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
