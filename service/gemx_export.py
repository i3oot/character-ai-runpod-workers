"""Export reconstructed SOMA joints without loading Blender or another AI model."""
import warnings
import numpy as np


def rotation_only(matrices):
    """Remove any fitted bone scale before exporting rotations."""
    u, _, vt = np.linalg.svd(matrices)
    sign = np.linalg.det(u @ vt)
    u = u.copy()
    u[..., :, -1] *= sign[..., None]
    return u @ vt


def bvh_channels(positions, rotations, rest_positions, rest_rotations, parents):
    """World-axis T-pose BVH, retaining local translations of procedural joints.

    SOMA world joint rotations include bind orientations. Convert to world-axis
    deltas, then localize those deltas. Keeping all six channels reproduces FK
    even when a procedural joint translates relative to its animated parent.
    """
    from scipy.spatial.transform import Rotation
    positions, rotations = np.asarray(positions), rotation_only(rotations)
    rest_rotations = rotation_only(rest_rotations)
    delta = rotations @ np.swapaxes(rest_rotations, -1, -2)
    offsets = np.empty_like(rest_positions)
    local_rotations = np.empty_like(delta)
    translations = np.empty_like(positions)
    for joint, parent in enumerate(parents):
        if parent < 0:
            offsets[joint] = rest_positions[joint]
            translations[:, joint] = positions[:, joint] - offsets[joint]
            local_rotations[:, joint] = delta[:, joint]
        else:
            offsets[joint] = rest_positions[joint] - rest_positions[parent]
            inverse = np.swapaxes(delta[:, parent], -1, -2)
            translations[:, joint] = np.einsum('tij,tj->ti', inverse,
                positions[:, joint] - positions[:, parent]) - offsets[joint]
            local_rotations[:, joint] = inverse @ delta[:, joint]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Gimbal lock detected')
        # Uppercase XYZ = intrinsic BVH Xrotation Yrotation Zrotation order.
        angles = Rotation.from_matrix(local_rotations.reshape(-1, 3, 3)).as_euler('XYZ', degrees=True)
    angles = angles.reshape(*local_rotations.shape[:2], 3)
    # Remove gratuitous 360-degree wraps while retaining the exact rotation.
    angles = np.rad2deg(np.unwrap(np.deg2rad(angles), axis=0))
    return offsets * 100, translations * 100, angles


def write_bvh(path, names, parents, positions, rotations, rest_positions, rest_rotations, fps=30):
    import re
    names, parents = list(names), np.asarray(parents)
    if len(names) != len(parents) or list(np.flatnonzero(parents < 0)) != [0]:
        raise ValueError('BVH must have one root at index zero')
    if any(not re.fullmatch(r'[A-Za-z0-9_]+', n) for n in names) or len(set(names)) != len(names):
        raise ValueError('Invalid BVH joint names')
    if any(p >= j for j, p in enumerate(parents)):
        raise ValueError('Parents must precede their children')
    for value in (positions, rotations, rest_positions, rest_rotations):
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite motion output')
    offsets, translations, angles = bvh_channels(positions, rotations, rest_positions, rest_rotations, parents)
    lines, order = ['HIERARCHY'], []

    def emit(joint, depth):
        indent = '  ' * depth
        lines.extend([indent + ('ROOT ' if joint == 0 else 'JOINT ') + names[joint], indent + '{',
            indent + '  OFFSET ' + ' '.join(f'{v:.8f}' for v in offsets[joint]),
            indent + '  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation'])
        order.append(joint)
        children = np.flatnonzero(parents == joint)
        for child in children:
            emit(int(child), depth + 1)
        if not len(children):
            lines.extend([indent + '  End Site', indent + '  {', indent + '    OFFSET 0 0 0', indent + '  }'])
        lines.append(indent + '}')

    emit(0, 0)
    lines.extend(['MOTION', f'Frames: {len(positions)}', f'Frame Time: {1 / fps:.10f}'])
    channels = np.concatenate((translations[:, order], angles[:, order]), axis=-1)
    lines.extend(' '.join(f'{v:.8f}' for v in frame.ravel()) for frame in channels)
    path.write_text('\n'.join(lines) + '\n')


def export_motion(result, soma_assets, folder):
    import torch
    from gem.common.soma import SomaFixedGs1BodyAdapter, SomaRuntimeConfig
    from gem.motion.postprocess.api import ieee_geometry
    parameters = result['pred_soma_params_global']
    frames = int(torch.as_tensor(result['length']).item())
    if not 15 <= frames <= 300 or parameters['body_pose'].shape[:2] != (1, frames):
        raise ValueError('Unexpected or unbounded GEM-X sequence length')
    adapter = SomaFixedGs1BodyAdapter(SomaRuntimeConfig(data_root=soma_assets, device='cuda',
        lod='xlo', mode='warp', apply_correctives=False, enable_procedural_transforms=True,
        procedural_reference='runtime'))
    transforms = []
    with ieee_geometry():
        for start in range(0, frames, 30):
            chunk = {key: (value[:, start:start + 30] if value.shape[:2] == (1, frames) else value).to('cuda')
                     for key, value in parameters.items()}
            geometry = adapter(**chunk, return_joints_only=True)
            transforms.append(geometry['transforms'][0, :, 1:].cpu().numpy())
    transforms = np.concatenate(transforms)
    runtime = adapter.runtime
    names = np.asarray(runtime.public_joint_names[1:])
    parents = runtime.output_joint_parent_ids.cpu().numpy()[1:] - 1
    bind = runtime.public_bind_transforms_world()[0, 1:].cpu().numpy()
    positions, rotations = transforms[..., :3, 3], rotation_only(transforms[..., :3, :3])
    rest_positions, rest_rotations = bind[..., :3, 3], rotation_only(bind[..., :3, :3])
    if len(names) != 77 or positions.shape != (frames, 77, 3):
        raise ValueError('Expected the public SOMA 77-joint skeleton')
    write_bvh(folder / 'motion.bvh', names, parents, positions, rotations, rest_positions, rest_rotations)
    # Pure numeric/unicode arrays: clients can use np.load(..., allow_pickle=False).
    values = dict(positions=positions, global_rot_mats=rotations, rest_positions=rest_positions,
        rest_rot_mats=rest_rotations, joint_names=names, parents=parents, fps=np.asarray(30),
        coordinate_system=np.asarray('Y_up_meters'))
    for key, value in parameters.items():
        values['soma_' + key] = value.cpu().numpy()
    np.savez_compressed(folder / 'motion.npz', **values)
    return {'positions': positions, 'names': names, 'parents': parents}


def make_preview(clip, result, geometry, folder):
    import cv2
    from gem.inference.video import VideoFrames
    from gem.inference.render import RGBVideoWriter
    cv2.setNumThreads(2)
    joints, parents = geometry['positions'], geometry['parents']
    points = result['conditions']['kp2d_native77'][0].cpu().numpy()
    bounds_lo, bounds_hi = joints.min(axis=(0, 1)), joints.max(axis=(0, 1))
    center = (bounds_lo + bounds_hi) / 2
    scale = min(330 / max(float(bounds_hi[0] - bounds_lo[0]), 0.5),
                390 / max(float(bounds_hi[1] - bounds_lo[1]), 0.5))
    with VideoFrames(clip) as frames, RGBVideoWriter(folder / 'preview.mp4', 960, 540, 30) as writer:
        if len(frames) != len(joints):
            raise ValueError('Preview clock differs from exported motion')
        for i in range(len(joints)):
            image = np.full((540, 960, 3), (23, 27, 35), dtype=np.uint8)
            frame = frames[i]
            ratio = min(470 / frames.width, 470 / frames.height)
            w, h = max(1, round(frames.width * ratio)), max(1, round(frames.height * ratio))
            x, y = (480 - w) // 2, (540 - h) // 2
            image[y:y + h, x:x + w] = cv2.resize(frame, (w, h))
            observed = points[i, :, :2] * ratio + [x, y]
            # Display fixed front view in world coordinates; never recenter each frame.
            projected = joints[i][:, [0, 1]] - center[[0, 1]]
            projected *= [scale, -scale]
            projected += [720, 285]
            for child, parent in enumerate(parents):
                if parent < 0:
                    continue
                cv2.line(image, tuple(np.rint(projected[parent]).astype(int)),
                    tuple(np.rint(projected[child]).astype(int)), (80, 210, 240), 2, cv2.LINE_AA)
                if points[i, child, 2] > 0.05 and points[i, parent, 2] > 0.05:
                    cv2.line(image, tuple(np.rint(observed[parent]).astype(int)),
                        tuple(np.rint(observed[child]).astype(int)), (100, 240, 140), 1, cv2.LINE_AA)
            for label, location in [('VIDEO + 2D ESTIMATE', (20, 28)),
                                    ('SOMA / WORLD FRONT VIEW', (500, 28)),
                                    (f'{i + 1}/{len(joints)}  |  30 fps', (500, 518))]:
                cv2.putText(image, label, location, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (225, 230, 235), 1, cv2.LINE_AA)
            writer.write(image)
            if i == len(joints) // 2:
                cv2.imwrite(str(folder / 'preview.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
