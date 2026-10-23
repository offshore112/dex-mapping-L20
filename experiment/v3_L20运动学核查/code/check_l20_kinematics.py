"""Check L20 FK against yourdfpy, export named cases and an offline viewer."""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from l20_kinematics import HandURDF, fingerprint, transform_points

ROOT = Path(__file__).resolve().parents[1]
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cases_for(hand, config):
    zero = dict.fromkeys(hand.independent, 0.)
    cases = [{'id': 'zero', 'q_independent': zero.copy()}]
    for name in hand.independent:
        cases.append({'id': f'single_{name}', 'q_independent': {**zero, name: config['single_joint_delta_rad']}})
        for i, bound in enumerate(hand.bounds[name]):
            cases.append({'id': f'bound_{name}_{i}', 'q_independent': {**zero, name: bound}})
    rng = np.random.default_rng(config['random_seed'])
    for i in range(config['random_configuration_count']):
        cases.append({'id': f'random_{i:03}', 'q_independent': {
            n: float(rng.uniform(*hand.bounds[n])) for n in hand.independent}})
    for joint in hand.joints:
        if joint.mimic:
            name, _, _ = hand.affine(joint.name)
            lo, hi = hand.bounds[name]
            for i, fraction in enumerate(config['coupled_sweep_fraction']):
                cases.append({'id': f'mimic_{joint.name}_{i}',
                              'q_independent': {**zero, name: lo+(hi-lo)*fraction}})
    return cases


def expected_rejection(hand, config):
    try:
        hand.expand(config)
    except ValueError:
        return True
    return False


def preview(path, hand, visuals, bounds, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(13, 8))
    zero, _, _ = hand.fk()
    states = [({}, 'URDF zero configuration'), ({'index_pip': .5}, 'index PIP = 0.5 rad; DIP = 0.44585 rad')]
    for i, (cfg, title) in enumerate(states):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        transforms, frames, _ = hand.fk(cfg)
        affected = hand.descendants(hand.joint_map['index_pip'].child)
        for visual in visuals:
            pts = transform_points(transforms[visual['link']], visual['triangles'].reshape(-1, 3)).reshape(-1, 3, 3)*1000
            color = '#e9a461' if i and visual['link'] in affected else '#a5bccb'
            ax.add_collection3d(Poly3DCollection(pts, facecolor=color, edgecolor='none', alpha=.55))
        for joint in hand.joints:
            p, q = (transforms[n][:3, 3]*1000 for n in (joint.parent, joint.child))
            ax.plot(*np.stack((p, q)).T, color='#264b67', linewidth=1.1)
            ax.scatter(*q, color='#264b67', s=8)
            if i:
                p0, q0 = (zero[n][:3, 3]*1000 for n in (joint.parent, joint.child))
                ax.plot(*np.stack((p0, q0)).T, color='#8c9ca6', linewidth=.7, linestyle='--')
        for axis, color, label in zip(np.eye(3), ['#cb4049','#31874c','#3173bb'], ['+X', '+Y', '+Z']):
            ax.quiver(0, 0, 0, *(axis*38), color=color, arrow_length_ratio=.15)
            ax.text(*(axis*42), label, color=color)
        frame = frames['index_pip']
        axis = frame[:3, :3] @ hand.joint_map['index_pip'].axis
        ax.quiver(*(frame[:3,3]*1000), *(axis*38), color='#bb388d', linewidth=2)
        mid = np.mean(bounds, axis=0)*1000
        span = np.max(np.ptp(bounds, axis=0))*1000
        for setter, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), mid):
            setter(value-span*.58, value+span*.58)
        ax.set_box_aspect([1,1,1])
        ax.view_init(elev=12, azim=20)
        ax.set_xlabel('base X (mm)'); ax.set_ylabel('base Y (mm)'); ax.set_zlabel('base Z (mm)')
        ax.set_title(title, fontsize=12, pad=20)
    fig.suptitle('L20 LEFT | root = hand_base_link | magenta = positive index PIP axis', fontsize=14)
    fig.text(.5,.04,'Sampled visual meshes. Grey dashed = zero skeleton. Link origins are not fingertips. Hardware has not been checked.',ha='center',fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path, default=ROOT/'environment/mano_l20_paths.json')
    parser.add_argument('--config', type=Path, default=ROOT/'experiment/v3_L20运动学核查/config.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = json.loads(args.paths.read_text(encoding='utf-8'))
    config = json.loads(args.config.read_text(encoding='utf-8'))
    out = args.output.resolve()
    if (out/'metrics.json').exists():
        raise FileExistsError('Preserve existing results: use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out/'code'; snapshot.mkdir(exist_ok=True)
    for name in ['l20_kinematics.py','check_l20_kinematics.py','l20_fk_viewer_template.html']:
        shutil.copyfile(ROOT/'scripts'/name, snapshot/name)
    write_json(out/'run_config.json', config)
    write_json(out/'input_paths.json', paths)
    hand = HandURDF(paths['l20_left_urdf'])
    assert hand.root == config['root_frame'] and paths['side'] == config['side'] == 'left'
    cases = cases_for(hand, config)
    write_json(out/'cases.json', cases)
    from yourdfpy import URDF
    reference = URDF.load(str(hand.path), load_meshes=False, build_scene_graph=True)
    assert set(reference.actuated_joint_names) == set(hand.independent)
    zero, zero_frames, _ = hand.fk()
    errors, references, mimic_rows = [], [], []
    max_orthogonal = max_determinant = max_bottom = max_mimic = 0.
    bone_values = {f'{f}_{bone}': [] for f in FINGERS[1:] for bone in ('proximal','middle')}
    for case in cases:
        transforms, frames, q = hand.fk(case['q_independent'])
        # Reference receives only independent names; its own parser expands mimic.
        reference.update_cfg(case['q_independent'])
        position_error = orientation_error = 0.
        for link, matrix in transforms.items():
            other = reference.get_transform(link, hand.root)
            position_error = max(position_error, float(np.linalg.norm(matrix[:3,3]-other[:3,3])))
            orientation_error = max(orientation_error, float(Rotation.from_matrix(matrix[:3,:3].T @ other[:3,:3]).magnitude()))
            r = matrix[:3,:3]
            max_orthogonal = max(max_orthogonal, float(np.max(np.abs(r.T@r-np.eye(3)))))
            max_determinant = max(max_determinant, abs(float(np.linalg.det(r))-1))
            max_bottom = max(max_bottom, float(np.max(np.abs(matrix[3]-[0,0,0,1]))))
        for joint in hand.joints:
            if joint.mimic:
                expected = joint.mimic['multiplier']*q[joint.mimic['joint']]+joint.mimic['offset']
                residual = abs(q[joint.name]-expected)
                max_mimic = max(max_mimic, residual)
                mimic_rows.append({'case_id':case['id'],'follower':joint.name,'leader':joint.mimic['joint'],
                    'leader_rad':q[joint.mimic['joint']],'follower_rad':q[joint.name],'residual_rad':residual})
        for finger in FINGERS[1:]:
            p = [frames[f'{finger}_{name}'][:3,3] for name in ('mcp_pitch','pip','dip')]
            for label, value in [('proximal',np.linalg.norm(p[1]-p[0])),('middle',np.linalg.norm(p[2]-p[1]))]:
                bone_values[f'{finger}_{label}'].append(float(value))
        errors.append({'case_id':case['id'],'max_link_position_error_m':position_error,'max_link_orientation_error_rad':orientation_error})
        references.append({'id':case['id'],'q_independent':case['q_independent'],
                           'transforms':{name:t.tolist() for name,t in transforms.items()}})
    isolation = derivative_errors = origin_motion = 0.
    for name in hand.independent:
        joint = hand.joint_map[name]
        moved, _, _ = hand.fk({name:config['single_joint_delta_rad']})
        affected = hand.descendants(joint.child)
        isolation = max(isolation, max(float(np.max(np.abs(t-zero[link]))) for link,t in moved.items() if link not in affected))
        origin_motion = max(origin_motion,float(np.linalg.norm(moved[joint.child][:3,3]-zero[joint.child][:3,3])))
        mid = {n:float(np.mean(hand.bounds[n])) for n in hand.independent}
        t, f, _ = hand.fk(mid)
        probe = np.array([0.,.02,0.]) if abs(joint.axis[0])>.8 else np.array([.02,0.,0.])
        h = config['finite_difference_step_rad']
        plus, _, _ = hand.fk({**mid,name:mid[name]+h})
        minus, _, _ = hand.fk({**mid,name:mid[name]-h})
        numerical = (transform_points(plus[joint.child],probe)-transform_points(minus[joint.child],probe))/(2*h)
        point = transform_points(t[joint.child],probe)
        expected = np.cross(f[name][:3,:3]@joint.axis, point-f[name][:3,3])
        derivative_errors = max(derivative_errors,float(np.linalg.norm(numerical-expected)))
    name = hand.independent[0]
    rejections = {'unknown_joint':expected_rejection(hand,{'unknown_joint':0}),
                  'passive_joint_command':expected_rejection(hand,{'index_dip':.1}),
                  'out_of_range':expected_rejection(hand,{name:hand.bounds[name][1]+.1})}
    lengths = {name:{'reference_mm':values[0]*1000,'max_variation_m':float(np.ptp(values))} for name,values in bone_values.items()}
    visuals, geometry_summary = hand.geometry(config['viewer_triangles_per_visual'])
    all_vertices = np.concatenate([transform_points(zero[v['link']],v['vertices_full']) for v in visuals])
    bounds = np.stack((all_vertices.min(0),all_vertices.max(0)))
    example_t, example_f, example_q = hand.fk({'index_pip':.5})
    joints = [j.record() for j in hand.ordered]
    scene = {'root':hand.root,'joints':joints,'independent':hand.independent,'bounds':hand.bounds,
             'center':bounds.mean(0).tolist(),'span':float(np.max(np.ptp(bounds,axis=0))),
             'visuals':[{'link':v['link'],'triangles':np.round(v['triangles'],9).tolist()} for v in visuals]}
    template = (ROOT/'scripts/l20_fk_viewer_template.html').read_text(encoding='utf-8')
    (out/'l20_fk_viewer.html').write_text(template.replace('__SCENE_DATA__',json.dumps(scene,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')
    write_json(out/'fk_reference_cases.json',references)
    write_json(out/'zero_configuration.json',{'root':hand.root,'unit':'metre/radian','independent_order':hand.independent,
        'q':hand.expand(),'link_transforms':{n:t.tolist() for n,t in zero.items()},
        'joint_axis_base':{j.name:(zero_frames[j.name][:3,:3]@j.axis).tolist() for j in hand.joints}})
    rows = []
    for joint in hand.joints:
        p = zero_frames[joint.name][:3,3]*1000
        rows.append({'joint':joint.name,'parent':joint.parent,'child':joint.child,'type':joint.kind,
            'independent':joint.name in hand.independent,'lower_rad':joint.lower,'upper_rad':joint.upper,
            'axis_local':str(joint.axis.tolist()),'origin_base_mm':str(p.tolist()),
            'axis_base_at_zero':str((zero_frames[joint.name][:3,:3]@joint.axis).tolist()),
            'mimic_leader':joint.mimic['joint'] if joint.mimic else '',
            'mimic_multiplier':joint.mimic['multiplier'] if joint.mimic else ''})
    write_csv(out/'joint_table.csv',rows);write_csv(out/'per_case_errors.csv',errors);write_csv(out/'mimic_sweeps.csv',mimic_rows)
    write_json(out/'geometry_audit.json',{'visuals':geometry_summary,'zero_bounds_base_m':bounds.tolist(),
        'zero_extents_base_mm':(np.ptp(bounds,axis=0)*1000).tolist(),'physical_scale_calibrated':False})
    write_json(out/'coordinate_notes.json',{'root':hand.root,'length_unit':'metre','angle_unit':'radian',
        'rpy':'Rz(yaw) @ Ry(pitch) @ Rx(roll)','joint_axis_frame':'joint frame after origin; not global axes',
        'positive_rotation':'right-hand rule','anatomical_wrist_defined':False,'tip_points_defined':False,
        'index_pip_at_zero_base_mm':(zero_frames['index_pip'][:3,3]*1000).tolist(),
        'index_pip_example_q':{n:q for n,q in example_q.items() if q!=0},
        'index_dip_origin_displacement_example_mm':((example_f['index_dip'][:3,3]-zero_frames['index_dip'][:3,3])*1000).tolist()})
    preview(out/'l20_fk_comparison.png',hand,visuals,bounds,config)
    max_position = max(r['max_link_position_error_m'] for r in errors)
    max_orientation = max(r['max_link_orientation_error_rad'] for r in errors)
    checks = {'structure_counts':len(hand.links)==22 and len(hand.joints)==21 and len(hand.independent)==16,
        'reference_position':max_position<=config['position_tolerance_m'],
        'reference_orientation':max_orientation<=config['orientation_tolerance_rad'],
        'proper_rotation':max(max_orthogonal,max_determinant,max_bottom)<=config['matrix_tolerance'],
        'subtree_isolation':isolation<=config['position_tolerance_m'],
        'joint_origin_stationary_under_own_rotation':origin_motion<=config['position_tolerance_m'],
        'positive_axis_derivative':derivative_errors<=config['axis_derivative_tolerance_m_per_rad'],
        'mimic_relation':max_mimic<=config['limit_tolerance_rad'],
        'bone_length_invariant':max(v['max_variation_m'] for v in lengths.values())<=config['position_tolerance_m'],
        'reject_invalid_inputs':all(rejections.values())}
    metrics = {'version':config['version'],'date':config['date'],'status':'passed' if all(checks.values()) else 'failed',
        'scope':'offline URDF kinematic consistency; no physical calibration/dynamics/hardware',
        'source':fingerprint(hand.path),'config':config,'sample_count':len(cases),'link_count':len(hand.links),
        'joint_count':len(hand.joints),'independent_coordinate_count':len(hand.independent),
        'mimic_count':sum(j.mimic is not None for j in hand.joints),'checks':checks,
        'max_reference_position_error_m':max_position,'max_reference_orientation_error_rad':max_orientation,
        'max_orthogonal_error':max_orthogonal,'max_determinant_error':max_determinant,
        'max_subtree_leakage_matrix_abs':isolation,'max_joint_origin_motion_m':origin_motion,
        'max_axis_derivative_error_m_per_rad':derivative_errors,'max_mimic_residual_rad':max_mimic,
        'bone_lengths':lengths,'invalid_inputs_rejected':rejections,
        'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','trimesh','yourdfpy']},
        'physical_scale_calibrated':False,'hardware_coupling_verified':False,
        'anatomical_wrist_and_tip_points_defined':False,'collision_checked':False,'retargeting_run':False}
    write_json(out/'metrics.json',metrics)
    print(json.dumps({'status':metrics['status'],'cases':len(cases),'reference_position_error_m':max_position,
                      'reference_orientation_error_rad':max_orientation,'checks':checks},ensure_ascii=True))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    sys.exit(main())
