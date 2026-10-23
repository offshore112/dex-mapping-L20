"""Load a legacy MANO model into smplx and export a reproducible CPU check.

No training or robot fitting is performed. Original model files are read-only.
The compatibility reader supports stored Ch/Select arrays only, not arbitrary
Chumpy graphs. Joint/tip conventions follow the author's smplx implementation:
https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import csc_matrix


class StoredCh:
    """Passive holder for a serialized Chumpy array's state."""


class StoredSelect:
    """Passive holder for a serialized Chumpy selection's state."""


class ManoUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('scipy.sparse.csc', 'csc_matrix'): csc_matrix,
            ('scipy.sparse._csc', 'csc_matrix'): csc_matrix,
            ('__builtin__', 'set'): set,
            ('builtins', 'set'): set,
            ('chumpy.ch', 'Ch'): StoredCh,
            ('chumpy.reordering', 'Select'): StoredSelect,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported MANO pickle class: {module}.{name}')
        return allowed[(module, name)]


def stored_array(value):
    if isinstance(value, StoredCh):
        return np.asarray(value.x)
    if isinstance(value, StoredSelect):
        selected = stored_array(value.a).reshape(-1)[value.idxs]
        return selected.reshape(value.preferred_shape)
    return np.asarray(value)


def read_mano(path):
    with path.open('rb') as stream:
        data = ManoUnpickler(stream, encoding='latin1').load()
    for key, value in list(data.items()):
        if isinstance(value, (StoredCh, StoredSelect)):
            data[key] = stored_array(value)
    expected = {'v_template': (778, 3), 'f': (1538, 3),
                'shapedirs': (778, 3, 10), 'posedirs': (778, 3, 135),
                'weights': (778, 16), 'kintree_table': (2, 16),
                'hands_mean': (45,), 'hands_components': (45, 45)}
    for key, shape in expected.items():
        if np.shape(data[key]) != shape:
            raise ValueError(f'{key}: expected {shape}, got {np.shape(data[key])}')
        if not np.isfinite(data[key]).all():
            raise ValueError(f'{key} contains non-finite values')
    if data['J_regressor'].shape != (16, 778):
        raise ValueError('Unexpected joint regressor shape')
    if data['f'].min() < 0 or data['f'].max() >= 778:
        raise ValueError('Invalid triangle vertex index')
    if not np.allclose(data['weights'].sum(axis=1), 1, atol=1e-7):
        raise ValueError('Skinning weights do not sum to one')
    return data


# Native MANO order: wrist, index, middle, pinky, ring, thumb (3 per finger).
CHAINS = {'thumb': [13, 14, 15], 'index': [1, 2, 3],
          'middle': [4, 5, 6], 'ring': [10, 11, 12], 'pinky': [7, 8, 9]}
TIP_VERTEX_IDS = {'thumb': 744, 'index': 320, 'middle': 443, 'ring': 554, 'pinky': 671}


def canonical_landmarks(vertices, joints):
    points, names = [joints[0]], ['wrist']
    for finger, indices in CHAINS.items():
        names.extend([f'{finger}_base', f'{finger}_middle', f'{finger}_distal', f'{finger}_tip'])
        points.extend([joints[index] for index in indices])
        points.append(vertices[TIP_VERTEX_IDS[finger]])
    return np.stack(points), names


def source_audit(path):
    return {'path': str(path.resolve()), 'size_bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def audit_urdf(path):
    robot = ET.parse(path).getroot()
    joints = robot.findall('joint')
    refs = sorted({node.attrib['filename'] for node in robot.findall('.//mesh')})
    meshes = []
    for ref in refs:
        if ref.startswith(('package://', 'file://')):
            raise ValueError('This checker expects the L20 model relative mesh paths')
        mesh_path = (path.parent / ref).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)
        meshes.append(source_audit(mesh_path))
    return {'source': source_audit(path), 'robot_name': robot.attrib['name'],
            'links': len(robot.findall('link')), 'joints': len(joints),
            'mimics': sum(j.find('mimic') is not None for j in joints),
            'mesh_files': meshes, 'all_mesh_files_present': True,
            'fk_checked': False, 'mesh_scale_checked': False,
            'hardware_driver_mapping_checked': False}


def export_obj(path, vertices, faces):
    with path.open('w', encoding='utf-8') as stream:
        stream.write('# MANO left; beta=0, full local pose=0; native coordinates\n')
        for vertex in vertices:
            stream.write('v ' + ' '.join(f'{x:.10f}' for x in vertex) + '\n')
        for face in faces:
            stream.write('f ' + ' '.join(str(int(x) + 1) for x in face) + '\n')


def preview(path, vertices, faces, points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    colors = ['#cf7441', '#387ba9', '#539173', '#986ba6', '#be617d']
    # PCA defines a display-only palm plane; saved data stay in native coordinates.
    center = vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(vertices - center, full_matrices=False)
    if np.linalg.det(axes) < 0:
        axes[-1] *= -1  # Keep the left hand under a proper display rotation.
    display = (vertices - center) @ axes.T * 1000
    markers = (points - center) @ axes.T * 1000
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': '3d'})
    mesh = Poly3DCollection(display[faces], facecolor='#cbd8e3',
                            edgecolor='#8896a2', linewidth=0.1, alpha=0.38)
    ax.add_collection3d(mesh)
    for i, (finger, color) in enumerate(zip(CHAINS, colors)):
        ids = [0] + list(range(1 + 4*i, 5 + 4*i))
        chain = markers[ids]
        ax.plot(*chain.T, color=color, linewidth=2, marker='o', markersize=4, label=finger)
        ax.scatter(*chain[-1], color=color, marker='^', s=65)
    ax.scatter(*markers[0], color='#263b50', s=65)
    span = max(np.ptp(display, axis=0))
    mid = (display.max(axis=0) + display.min(axis=0)) / 2
    for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
        setter(value-span*.55, value+span*.55)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=75, azim=-85)
    ax.set_xlabel('Display axis 1 (mm)')
    ax.set_ylabel('Display axis 2 (mm)')
    ax.set_zlabel('Display axis 3 (mm)')
    ax.set_title('MANO LEFT | neutral shape and flat reference pose\n'
                 '21 landmarks: wrist + 5 x (3 joints + tip vertex)', pad=16)
    ax.legend(loc='upper left', fontsize=9)
    fig.text(.5, .03, 'Preview only. Metre convention; robot geometry fitting has not been run.',
             ha='center', fontsize=9, color='#4e6275')
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config['side'] != 'left':
        raise ValueError('This check is configured for the left hand only')
    model_path = Path(config['mano_left'])
    data = read_mano(model_path)
    import torch
    import smplx
    from smplx.utils import Struct
    torch.set_num_threads(1)
    dtype = torch.float64
    model = smplx.MANO(str(model_path), data_struct=Struct(**data),
                       is_rhand=False, use_pca=False, flat_hand_mean=True,
                       num_betas=10, dtype=dtype)
    params = {'betas': torch.zeros(1, 10, dtype=dtype),
              'hand_pose': torch.zeros(1, 45, dtype=dtype),
              'global_orient': torch.zeros(1, 3, dtype=dtype),
              'transl': torch.zeros(1, 3, dtype=dtype)}
    result = model(**params)
    vertices = result.vertices[0].detach().numpy()
    joints = result.joints[0, :16].detach().numpy()
    neutral_error = float(np.max(np.abs(vertices - data['v_template'])))
    if neutral_error > 1e-9 or not np.isfinite(vertices).all():
        raise ValueError(f'Neutral forward does not reproduce template: {neutral_error}')
    joint_error = float(np.max(np.abs(joints - data['J_regressor'].dot(data['v_template']))))
    if joint_error > 1e-9:
        raise ValueError(f'Joint regressor mismatch: {joint_error}')
    shape = params['betas'].clone().requires_grad_(True)
    loss = model(**{**params, 'betas': shape}).vertices.square().sum()
    loss.backward()
    if not torch.isfinite(shape.grad).all() or shape.grad.abs().sum() == 0:
        raise ValueError('MANO shape gradient check failed')
    # This perturbation checks the pose forward path; it is not a grasp label.
    posed = params['hand_pose'].clone()
    posed[0, 3:6] = torch.tensor([0.2, 0, 0], dtype=dtype)
    posed_vertices = model(**{**params, 'hand_pose': posed}).vertices[0].detach().numpy()
    pose_change = float(np.max(np.linalg.norm(posed_vertices-vertices, axis=1)))
    if not np.isfinite(posed_vertices).all() or pose_change <= 0:
        raise ValueError('MANO pose perturbation check failed')
    points, names = canonical_landmarks(vertices, joints)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out/'environment_check.json'
    if report_path.exists():
        raise FileExistsError('Preserve previous run: choose a new output directory')
    report = {'status': 'passed_model_environment_check_only',
              'python': sys.version, 'executable': sys.executable,
              'packages': {name: importlib.metadata.version(name) for name in
                           ('torch', 'numpy', 'scipy', 'matplotlib', 'smplx', 'trimesh', 'Markdown')},
              'cuda_available': torch.cuda.is_available(),
              'mano_source': source_audit(model_path),
              'mano_shapes': {key: list(np.shape(data[key])) for key in
                              ('v_template', 'f', 'shapedirs', 'posedirs', 'weights')},
              'mano_joint_count_native': 16, 'landmark_count_exported': 21,
              'landmark_order': names, 'tip_vertex_ids': TIP_VERTEX_IDS,
              'tip_convention_source': 'https://github.com/vchoutas/smplx/blob/main/smplx/vertex_ids.py',
              'neutral_vertices_max_abs_error_model_units': neutral_error,
              'neutral_joints_max_abs_error_model_units': joint_error,
              'shape_gradient_finite_and_nonzero': True,
              'pose_perturbation_max_vertex_displacement_model_units': pose_change,
              'coordinate_unit': config['model_coordinate_unit'],
              'coordinates': 'MANO native axes; preview alone uses PCA display axes',
              'l20_resources': audit_urdf(Path(config['l20_left_urdf'])),
              'robot_geometry_fitting_run': False, 'training_run': False}
    export_obj(out/'mano_neutral.obj', vertices, data['f'])
    np.savez(out/'mano_reference.npz', vertices=vertices, faces=data['f'],
             joints_native16=joints, landmarks21=points, betas=params['betas'].numpy(),
             full_local_pose=params['hand_pose'].numpy())
    (out/'landmarks_mano21.json').write_text(json.dumps({
        'side': 'left', 'coordinate_frame': 'MANO native',
        'unit': config['model_coordinate_unit'], 'tip_vertex_ids': TIP_VERTEX_IDS,
        'landmarks': [{'index': i, 'name': name, 'xyz': point.tolist()}
                      for i, (name, point) in enumerate(zip(names, points))]}, indent=2), encoding='utf-8')
    preview(out/'mano_left_neutral.png', vertices, data['f'], points)
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': str(out),
                      'vertices': len(vertices), 'native_joints': 16, 'landmarks': 21,
                      'torch': report['packages']['torch']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
