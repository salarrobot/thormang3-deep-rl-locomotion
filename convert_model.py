from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np
import xacro
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent
DESC = ROOT / "external/ROBOTIS-THORMANG-Common/thormang3_description"
BUILD = ROOT / "build"
ASSETS = ROOT / "assets"
SCENE_XML = ASSETS / "thormang3_scene.xml"

LEG_JOINTS = [
    "l_leg_hip_y", "l_leg_hip_r", "l_leg_hip_p",
    "l_leg_kn_p", "l_leg_an_p", "l_leg_an_r",
    "r_leg_hip_y", "r_leg_hip_r", "r_leg_hip_p",
    "r_leg_kn_p", "r_leg_an_p", "r_leg_an_r",
]

WELD_POSE = {
    "l_arm_sh_p1": 0.90, "r_arm_sh_p1": -0.90,
    "l_arm_sh_r": 1.50, "r_arm_sh_r": -1.50,
}

CROUCH = 0.35
KP = 400.0
FORCERANGE = 44.0
JOINT_DAMPING = 2.0
JOINT_ARMATURE = 0.1
JOINT_FRICTIONLOSS = 0.2
TIMESTEP = 0.0025

_PLACEHOLDER_PAIR = re.compile(
    r'<inertia\s+ixx="1\.0"[^>]*?/>\s*<!--\s*(<inertia[^>]*?/>)\s*-->', re.S)
_PLACEHOLDER_LONE = re.compile(r'<inertia\s+ixx="1\.0"[^>]*?/>')
_LONE_FALLBACK = ('<inertia ixx="0.001" ixy="0.0" ixz="0.0" '
                  'iyy="0.001" iyz="0.0" izz="0.001" />')


def _fmt(vec) -> str:
    return " ".join(f"{v:.6g}" for v in np.asarray(vec).flatten())


def _parse(attr: str | None, default: str) -> np.ndarray:
    return np.fromstring(attr if attr else default, sep=" ")


def quat_about(axis: np.ndarray, angle: float) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, axis / np.linalg.norm(axis), angle)
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def rot_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, v, q)
    return out


def _sanitize_inertias(urdf: str) -> str:
    root = ET.fromstring(urdf)
    n_fixed = 0
    for inertia in root.iter("inertia"):
        g = lambda k: float(inertia.get(k))
        fixed = False
        for k in ("ixx", "iyy", "izz"):
            if g(k) < 1e-5:
                inertia.set(k, "1e-5")
                fixed = True
        M = np.array([[g("ixx"), g("ixy"), g("ixz")],
                      [g("ixy"), g("iyy"), g("iyz")],
                      [g("ixz"), g("iyz"), g("izz")]])
        if np.linalg.eigvalsh(M).min() <= 1e-9:
            for k in ("ixy", "ixz", "iyz"):
                inertia.set(k, "0.0")
            fixed = True
        n_fixed += fixed
    if n_fixed:
        print(f"[urdf] zeroed off-diagonal terms of {n_fixed} "
              "non-positive-definite inertia matrices")
    return ET.tostring(root, encoding="unicode")


def build_urdf() -> Path:
    BUILD.mkdir(exist_ok=True)
    n_restored = n_fallback = 0
    for src in (DESC / "urdf").iterdir():
        text = src.read_text()
        text = text.replace("$(find thormang3_description)/urdf/", "")
        text, k = _PLACEHOLDER_PAIR.subn(lambda m: m.group(1), text)
        n_restored += k
        text, k = _PLACEHOLDER_LONE.subn(_LONE_FALLBACK, text)
        n_fallback += k
        (BUILD / src.name).write_text(text)
    print(f"[urdf] restored {n_restored} real inertias, "
          f"{n_fallback} fallback inertias")

    doc = xacro.process_file(str(BUILD / "thormang3.xacro"))
    urdf = _sanitize_inertias(doc.toprettyxml(indent="  "))

    mesh_dir = ASSETS / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for stl in (DESC / "meshes").glob("*.stl"):
        shutil.copy2(stl, mesh_dir / stl.name)

    mujoco_block = (
        '\n  <mujoco>\n'
        f'    <compiler meshdir="{mesh_dir}" strippath="true" '
        'balanceinertia="true" discardvisual="false" fusestatic="false"/>\n'
        '  </mujoco>\n</robot>')
    urdf = urdf.replace("</robot>", mujoco_block)
    out = BUILD / "thormang3.urdf"
    out.write_text(urdf)
    print(f"[urdf] wrote {out}")
    return out


def urdf_to_raw_mjcf(urdf_path: Path) -> Path:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    raw = BUILD / "thormang3_raw.xml"
    mujoco.mj_saveLastXML(str(raw), model)
    print(f"[mjcf] URDF imported: nbody={model.nbody} njnt={model.njnt} "
          f"ngeom={model.ngeom} mass={mujoco.mj_getTotalmass(model):.2f} kg")
    return raw


def _weld_joint(body: ET.Element, joint: ET.Element, angle: float) -> None:
    axis = _parse(joint.get("axis"), "0 0 1")
    jpos = _parse(joint.get("pos"), "0 0 0")
    b_pos = _parse(body.get("pos"), "0 0 0")
    b_quat = _parse(body.get("quat"), "1 0 0 0")
    qj = quat_about(axis, angle)
    new_quat = quat_mul(b_quat, qj)
    new_pos = b_pos + rot_vec(b_quat, jpos - rot_vec(qj, jpos))
    body.set("pos", _fmt(new_pos))
    body.set("quat", _fmt(new_quat))
    body.remove(joint)


def postprocess(raw_path: Path) -> None:
    tree = ET.parse(raw_path)
    root = tree.getroot()
    root.set("model", "thormang3")

    comp = root.find("compiler")
    comp.attrib.clear()
    comp.set("angle", "radian")
    comp.set("meshdir", "meshes")
    comp.set("autolimits", "true")

    for tag in ("option", "visual", "asset", "actuator", "contact"):
        if root.find(tag) is None:
            root.append(ET.Element(tag))
    opt = root.find("option")
    opt.set("timestep", str(TIMESTEP))
    opt.set("integrator", "implicitfast")

    vis = root.find("visual")
    ET.SubElement(vis, "global", offwidth="1920", offheight="1080")
    ET.SubElement(vis, "quality", shadowsize="4096")
    ET.SubElement(vis, "headlight", ambient="0.4 0.4 0.4",
                  diffuse="0.5 0.5 0.5", specular="0.1 0.1 0.1")
    ET.SubElement(vis, "map", znear="0.01", shadowclip="3")

    asset = root.find("asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.45 0.62 0.82", rgb2="0.85 0.90 0.96",
                  width="512", height="3072")
    ET.SubElement(asset, "texture", type="2d", name="tex_floor",
                  builtin="checker", mark="edge",
                  rgb1="0.29 0.32 0.36", rgb2="0.24 0.27 0.31",
                  markrgb="0.75 0.78 0.82", width="300", height="300")
    ET.SubElement(asset, "material", name="mat_floor", texture="tex_floor",
                  texuniform="true", texrepeat="4 4", reflectance="0.15")

    for mesh in asset.findall("mesh"):
        mesh.set("file", Path(mesh.get("file")).name)

    world = root.find("worldbody")
    ET.SubElement(world, "light", name="sun", directional="true",
                  castshadow="true", pos="0 0 3", dir="0.15 0.2 -1",
                  diffuse="0.7 0.7 0.7", specular="0.2 0.2 0.2")
    ET.SubElement(world, "geom", name="floor", type="plane",
                  size="200 200 0.05", material="mat_floor",
                  friction="1.0 0.005 0.0001", contype="1", conaffinity="1")

    cam_pos = np.array([2.0, -3.4, 0.9])
    target = np.array([0.0, 0.0, 0.05])
    fwd = target - cam_pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    ET.SubElement(world, "camera", name="track", mode="trackcom",
                  pos=_fmt(cam_pos), xyaxes=_fmt(np.r_[right, up]))
    ET.SubElement(world, "camera", name="side", mode="trackcom",
                  pos="0 -3.2 0.35", xyaxes="1 0 0 0 0 1")

    pelvis = world.find("body")
    assert pelvis is not None and pelvis.get("name") == "pelvis_link"
    pelvis.insert(0, ET.Element("freejoint", name="root"))
    pelvis.set("pos", "0 0 0.85")
    ET.SubElement(pelvis, "light", name="spot", mode="trackcom",
                  pos="0 0 2.5", dir="0 0 -1", diffuse="0.35 0.35 0.35")

    bodies = pelvis.iter("body") if False else list(pelvis.iter("body")) + [pelvis]
    n_welded = 0
    for body in bodies:
        for joint in list(body.findall("joint")):
            name = joint.get("name")
            if name in LEG_JOINTS:
                joint.set("damping", str(JOINT_DAMPING))
                joint.set("armature", str(JOINT_ARMATURE))
                joint.set("frictionloss", str(JOINT_FRICTIONLOSS))
                joint.attrib.pop("actuatorfrcrange", None)
            else:
                _weld_joint(body, joint, WELD_POSE.get(name, 0.0))
                n_welded += 1
        bname = body.get("name", "")
        for geom in body.findall("geom"):
            if geom.get("type", "sphere") == "mesh":
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
                geom.set("group", "2")
            else:
                geom.set("group", "3")
                contype, conaffinity = 1, 0
                if bname.startswith("l_leg"):
                    conaffinity |= 2
                elif bname.startswith("r_leg"):
                    contype |= 2
                geom.set("contype", str(contype))
                geom.set("conaffinity", str(conaffinity))
                if bname.endswith("foot_link"):
                    geom.set("name", f"{bname[0]}_foot_col")
                    geom.set("friction", "1.0 0.005 0.0001")
    print(f"[mjcf] welded {n_welded} non-leg joints "
          f"(pose baked for {len(WELD_POSE)})")

    act = root.find("actuator")
    for jname in LEG_JOINTS:
        joint = next(j for j in pelvis.iter("joint") if j.get("name") == jname)
        ET.SubElement(act, "position", name=jname, joint=jname,
                      kp=str(KP), ctrlrange=joint.get("range"),
                      forcerange=f"-{FORCERANGE} {FORCERANGE}")

    ET.indent(tree, space="  ")
    tree.write(SCENE_XML, encoding="unicode")
    print(f"[mjcf] wrote {SCENE_XML}")


def _leg_stand_angles(model: mujoco.MjModel, side: str) -> dict[str, float]:
    def ysign(jname: str) -> float:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        return float(np.sign(model.jnt_axis[jid][1]))

    h, k, a = (ysign(f"{side}_leg_{j}") for j in ("hip_p", "kn_p", "an_p"))
    best = None
    for sh in (+1, -1):
        q_hip = sh * CROUCH
        q_kn = -2 * q_hip * h / k
        q_an = -(h * q_hip + k * q_kn) / a
        if h * q_hip < 0:
            best = {f"{side}_leg_hip_p": q_hip,
                    f"{side}_leg_kn_p": q_kn,
                    f"{side}_leg_an_p": q_an}
    assert best is not None
    return best


def add_keyframe_and_check() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)

    pose = {}
    for side in ("l", "r"):
        pose.update(_leg_stand_angles(model, side))

    def ysign(jname: str) -> float:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        return float(np.sign(model.jnt_axis[jid][1]))

    def fk(lean: float) -> tuple[np.ndarray, float, float, float]:
        data.qpos[:] = 0
        data.qpos[0:3] = [0, 0, 1.5]
        mujoco.mju_axisAngle2Quat(data.qpos[3:7], np.array([0, 1.0, 0]), lean)
        for jname, q in pose.items():
            data.qpos[model.joint(jname).qposadr[0]] = q
        for side in ("l", "r"):
            adr = model.joint(f"{side}_leg_an_p").qposadr[0]
            data.qpos[adr] -= lean / ysign(f"{side}_leg_an_p")
        mujoco.mj_forward(model, data)

        lowest, feet_x, flat = np.inf, [], 0.0
        for gname in ("l_foot_col", "r_foot_col"):
            gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            R = data.geom_xmat[gid].reshape(3, 3)
            flat = max(flat, abs(R[2, 0]), abs(R[2, 1]))
            half = model.geom_size[gid]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1)
                                for sy in (-1, 1) for sz in (-1, 1)]) * half
            z = (data.geom_xpos[gid] + corners @ R.T)[:, 2]
            lowest = min(lowest, z.min())
            feet_x.append(data.geom_xpos[gid][0])
        com = data.subtree_com[1]
        return data.qpos.copy(), com[0] - np.mean(feet_x), lowest, flat

    lean, com_h, margin = 0.0, 0.75, 0.010
    for _ in range(4):
        qpos, dx, lowest, flat = fk(lean)
        lean -= np.arctan2(dx - margin, com_h)
    qpos, dx, lowest, flat = fk(lean)
    assert flat < 0.005, f"feet not flat after lean (tilt={flat:.4f})"
    print(f"[stand] lean={np.degrees(lean):+.2f} deg, CoM offset "
          f"{dx * 1000:+.1f} mm ahead of foot centers")

    z0 = 1.5 - lowest + 0.002
    qpos[2] = z0
    ctrl = np.array([qpos[model.joint(j).qposadr[0]] for j in LEG_JOINTS])

    tree = ET.parse(SCENE_XML)
    root = tree.getroot()
    kf = ET.SubElement(root, "keyframe")
    ET.SubElement(kf, "key", name="stand", qpos=_fmt(qpos), ctrl=_fmt(ctrl))
    ET.indent(tree, space="  ")
    tree.write(SCENE_XML, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    up = np.zeros(3)
    mujoco.mju_rotVecQuat(up, np.array([0, 0, 1.0]), data.qpos[3:7])
    drift = np.linalg.norm(data.qpos[0:2])
    print(f"[stand] z0={z0:.3f}  after 3 s: z={data.qpos[2]:.3f} "
          f"xy-drift={drift:.3f} m  uprightness={up[2]:.4f}")
    ok = abs(data.qpos[2] - z0) < 0.05 and up[2] > 0.99 and drift < 0.05
    print(f"[stand] {'PASS' if ok else 'FAIL'}")


def render_png(path: str) -> None:
    import imageio.v3 as iio
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    for _ in range(400):
        mujoco.mj_step(model, data)
    with mujoco.Renderer(model, 720, 1280) as r:
        r.update_scene(data, camera="track")
        iio.imwrite(path, r.render())
    print(f"[render] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", help="also render a PNG of the stand keyframe")
    args = ap.parse_args()

    urdf = build_urdf()
    raw = urdf_to_raw_mjcf(urdf)
    postprocess(raw)
    add_keyframe_and_check()

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    print(f"[done] scene: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"ngeom={model.ngeom} mass={mujoco.mj_getTotalmass(model):.2f} kg")
    if args.png:
        render_png(args.png)


if __name__ == "__main__":
    main()
