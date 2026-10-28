import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from protocol import validate, identifier
from handler import validate_glb
import handler
import client


class ContractTests(unittest.TestCase):
    def test_cross_worker_and_unbounded_prompts(self):
        for prompt in ['A person walks. Then runs.', '', 'x' * 1001]:
            with self.assertRaises(ValueError):
                validate({'kind': 'kimodo', 'prompt': prompt}, 'kimodo')
        with self.assertRaises(ValueError):
            validate({'kind': 'kimodo', 'prompt': 'A person walks'}, 'skintokens')

    def test_motion_bounds(self):
        for duration in [float('nan'), float('inf'), -1, 11, True]:
            with self.assertRaises(ValueError):
                validate({'kind': 'kimodo', 'prompt': 'A person walks', 'duration': duration}, 'kimodo')
        validate({'kind': 'kimodo', 'prompt': 'A person walks.', 'duration': 3}, 'kimodo')

    def test_private_input_contract(self):
        request = {'kind': 'skintokens', 'mesh_key': 'character-ai/inputs/' + 'a' * 32 + '/mesh.glb',
                   'mesh_sha256': 'b' * 64, 'use_skeleton': True}
        validate(request, 'skintokens')
        for key in ['https://example.com/a.glb', '../../a.glb', 'pixal3d/inputs/a.glb']:
            with self.assertRaises(ValueError):
                validate(dict(request, mesh_key=key), 'skintokens')
        with self.assertRaises(ValueError):
            identifier('../x')

    def test_rigged_export_and_external_resource_validation(self):
        def write(path, doc):
            data = json.dumps(doc).encode()
            data += b' ' * (-len(data) % 4)
            path.write_bytes(struct.pack('<4sIII4s', b'glTF', 2, 20 + len(data), len(data), b'JSON') + data)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mesh.glb'
            write(path, {'meshes': [{}]})
            with self.assertRaises(ValueError):
                validate_glb(path, rigged=True)
            write(path, {'meshes': [{}], 'skins': [{'joints': [0]}]})
            validate_glb(path, rigged=True)
            write(path, {'meshes': [{}], 'buffers': [{'uri': '/etc/passwd'}]})
            with self.assertRaises(ValueError):
                validate_glb(path)

    def test_completed_job_does_not_reload_or_generate(self):
        result = {'artifacts': [], 'job_id': 'saved-job'}
        with patch.dict('os.environ', WORKER_KIND='kimodo'), \
                patch.object(handler, 's3_client'), patch.object(handler, 'bucket_name', return_value='private'), \
                patch.object(handler, 'read_result', return_value=result), \
                patch.object(handler, 'BACKEND', None):
            self.assertIs(handler.handler({'id': 'saved-job', 'input': {
                'kind': 'kimodo', 'prompt': 'A person walks.'}}), result)
            self.assertIsNone(handler.BACKEND)

    def test_fetch_uses_durable_storage_before_runpod(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ticket = root / 'job.json'
            client.save(ticket, {'endpoint_id': 'endpoint', 'job_id': 'job', 'input': {'kind': 'kimodo'},
                'metadata_key': 'character-ai/outputs/kimodo/job/result.json', 'bucket': 'private'})
            with patch.object(client, 's3_client'), \
                    patch.object(client, 'read_result', return_value={'artifacts': []}), \
                    patch.object(client, 'api', side_effect=AssertionError('Must not call RunPod')):
                client.fetch(ticket, root / 'result', 1, 1)
                self.assertTrue((root / 'result/result.json').is_file())

    def test_ambiguous_submission_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            ticket = Path(folder) / 'job.json'
            client.save(ticket, {'state': 'submitting'})
            with patch.object(client, 'api', side_effect=AssertionError('Must not resubmit')):
                with self.assertRaisesRegex(RuntimeError, 'Ambiguous submission'):
                    client.fetch(ticket, Path(folder), 1, 1)


if __name__ == '__main__':
    unittest.main()
