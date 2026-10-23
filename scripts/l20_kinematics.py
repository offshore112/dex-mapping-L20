"""L20 URDF tree kinematics in metres/radians; no hardware command interface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np


def vector(text, default):
    value = np.array(default if text is None else [float(x) for x in text.split()], dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f'Invalid vector: {text}')
    return value


def axis_rotation(axis, angle):
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)


def origin_matrix(node):
    xyz = vector(node.get('xyz') if node is not None else None, [0, 0, 0])
    rpy = vector(node.get('rpy') if node is not None else None, [0, 0, 0])
    r, p, y = rpy
    rotation = (axis_rotation([0, 0, 1], y) @ axis_rotation([0, 1, 0], p)
                @ axis_rotation([1, 0, 0], r))
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, xyz
    return matrix


def transform_points(matrix, points):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def fingerprint(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'size_bytes': path.stat().st_size}


@dataclass
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    mimic: dict | None

    def record(self):
        return {'name': self.name, 'type': self.kind, 'parent': self.parent,
                'child': self.child, 'origin': self.origin.tolist(),
                'axis': self.axis.tolist(), 'lower': self.lower, 'upper': self.upper,
                'mimic': self.mimic}


class HandURDF:
    def __init__(self, filename):
        self.path = Path(filename).resolve()
        self.xml = ET.parse(self.path).getroot()
        self.name = self.xml.attrib['name']
        link_nodes = self.xml.findall('link')
        self.links = [link.attrib['name'] for link in link_nodes]
        if len(set(self.links)) != len(self.links):
            raise ValueError('Duplicate link name')
        self.joints = []
        for element in self.xml.findall('joint'):
            kind = element.attrib['type']
            if kind not in ('revolute', 'fixed'):
                raise ValueError(f'Unsupported joint type: {kind}')
            axis_node, limit, mimic_node = element.find('axis'), element.find('limit'), element.find('mimic')
            axis = vector(axis_node.get('xyz') if axis_node is not None else None, [1, 0, 0])
            if abs(np.linalg.norm(axis)-1) > 1e-10:
                raise ValueError('Non-unit joint axis')
            mimic = None if mimic_node is None else {
                'joint': mimic_node.attrib['joint'],
                'multiplier': float(mimic_node.get('multiplier', 1)),
                'offset': float(mimic_node.get('offset', 0))}
            lower, upper = (0., 0.) if kind == 'fixed' else (float(limit.attrib['lower']), float(limit.attrib['upper']))
            if not np.isfinite([lower, upper]).all() or lower > upper:
                raise ValueError('Invalid joint limits')
            self.joints.append(Joint(element.attrib['name'], kind,
                element.find('parent').attrib['link'], element.find('child').attrib['link'],
                origin_matrix(element.find('origin')), axis, lower, upper, mimic))
        self.joint_map = {j.name: j for j in self.joints}
        if len(self.joint_map) != len(self.joints):
            raise ValueError('Duplicate joint name')
        children = [j.child for j in self.joints]
        if len(children) != len(set(children)):
            raise ValueError('A link has multiple parents')
        roots = set(self.links)-set(children)
        if len(roots) != 1:
            raise ValueError('Expected a single root')
        self.root = roots.pop()
        known, pending, self.ordered = {self.root}, self.joints.copy(), []
        while pending:
            ready = [j for j in pending if j.parent in known]
            if not ready:
                raise ValueError('Disconnected or cyclic joint tree')
            for j in ready:
                if j.child not in self.links:
                    raise ValueError('Missing child link')
                known.add(j.child)
                self.ordered.append(j)
                pending.remove(j)
        if known != set(self.links):
            raise ValueError('Disconnected links')
        self.independent = [j.name for j in self.joints if j.kind != 'fixed' and j.mimic is None]
        self.bounds = {name: [self.joint_map[name].lower, self.joint_map[name].upper] for name in self.independent}
        for j in self.joints:
            if not j.mimic:
                continue
            leader, a, b = self.affine(j.name)
            if abs(a) < 1e-15:
                if not j.lower <= b <= j.upper:
                    raise ValueError('Constant mimic violates limits')
                continue
            lower, upper = sorted(((j.lower-b)/a, (j.upper-b)/a))
            self.bounds[leader] = [max(self.bounds[leader][0], lower), min(self.bounds[leader][1], upper)]
            if self.bounds[leader][0] > self.bounds[leader][1]:
                raise ValueError('Mimic and joint limits have no common interval')

    def affine(self, name, visited=None):
        visited = set() if visited is None else visited
        if name in visited or name not in self.joint_map:
            raise ValueError('Cyclic or missing mimic leader')
        j = self.joint_map[name]
        if j.mimic is None:
            return name, 1., 0.
        leader, a, b = self.affine(j.mimic['joint'], visited | {name})
        m = j.mimic
        return leader, m['multiplier']*a, m['multiplier']*b+m['offset']

    def expand(self, config=None, tolerance=1e-12):
        config = config or {}
        bad = set(config)-set(self.independent)
        if bad:
            raise ValueError(f'Only independent joint names are accepted: {sorted(bad)}')
        values = {name: float(config.get(name, 0)) for name in self.independent}
        for name, value in values.items():
            lo, hi = self.bounds[name]
            if not np.isfinite(value) or not lo-tolerance <= value <= hi+tolerance:
                raise ValueError(f'Joint outside effective bounds: {name}={value}')
        full = {}
        for j in self.joints:
            if j.kind == 'fixed':
                full[j.name] = 0.
            else:
                leader, a, b = self.affine(j.name)
                full[j.name] = a*values[leader]+b
            if not j.lower-tolerance <= full[j.name] <= j.upper+tolerance:
                raise ValueError(f'Expanded mimic violates joint limits: {j.name}')
        return full

    def fk(self, config=None):
        full = self.expand(config)
        transforms, frames = {self.root: np.eye(4)}, {}
        for j in self.ordered:
            frame = transforms[j.parent] @ j.origin
            motion = np.eye(4)
            if j.kind != 'fixed':
                motion[:3, :3] = axis_rotation(j.axis, full[j.name])
            frames[j.name] = frame
            transforms[j.child] = frame @ motion
        return transforms, frames, full

    def descendants(self, child):
        found = {child}
        for j in self.ordered:
            if j.parent in found:
                found.add(j.child)
        return found

    def geometry(self, triangles_per_visual=1200):
        import trimesh
        visuals, summaries = [], []
        for link in self.xml.findall('link'):
            for visual in link.findall('visual'):
                node = visual.find('geometry/mesh')
                if node is None:
                    raise ValueError('This L20 checker expects mesh visuals')
                filename = node.attrib['filename']
                if '://' in filename:
                    raise ValueError('Expected a local relative mesh path')
                path = (self.path.parent/filename).resolve()
                mesh = trimesh.load_mesh(path, process=True)
                scale = vector(node.get('scale'), [1, 1, 1])
                local = transform_points(origin_matrix(visual.find('origin')), mesh.vertices*scale)
                ids = np.linspace(0, len(mesh.faces)-1, min(len(mesh.faces), triangles_per_visual), dtype=int)
                triangles = local[mesh.faces[ids]]
                visuals.append({'link': link.attrib['name'], 'triangles': triangles,
                                'vertices_full': local})
                summaries.append({'link': link.attrib['name'], **fingerprint(path),
                                  'mesh_scale': scale.tolist(), 'faces_full': len(mesh.faces),
                                  'faces_display': len(ids), 'bounds_link_m': [local.min(0).tolist(), local.max(0).tolist()]})
        return visuals, summaries
