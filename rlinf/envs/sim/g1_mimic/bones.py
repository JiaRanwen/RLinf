# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G1-retargeted BONES-SEED name and feasibility filters (Appendix E).

The paper starts from 142,220 clips, keeps 129,667 after the name filter
and 129,514 after the feasibility filter. Those counts belong to that
release. A different dump must be reported as a different dump. This module
does not download data and does not accept a human skeleton as a G1 clip.

The floating-feet gate in the paper measures MuJoCo geom-to-plane distance.
``feasibility_reject`` applies that gate to a clearance the caller computed;
``bones_geometry.G1FloorDistance`` computes it with ``mj_geomDistance``. A
link-origin height is not that distance. Upright is derived from the root
quaternion. A missing field is not treated as upright.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from rlinf.envs.sim.g1_mimic.kinematics.features import projected_gravity
from rlinf.envs.sim.g1_mimic.tasks.motion import (
    MotionLibrary,
    control_time_grid,
    decode_motion_payload,
    resample_series,
)
from rlinf.utils.logging import get_logger

# Project convention for the pelvis-upright gate. Identity has body-frame
# gravity z of -1. A 90 degree roll is about 0 and is not upright. This is
# not Appendix E's geom distance and not the velocity-fall tilt.
UPRIGHT_GRAVITY_Z = -0.7

# Terms named in Appendix E. Short words use a boundary so "visit" is kept.
_PHRASES = ("jump off 50cm", "look over fence", "stair", "ladder", "climb", "fence")
_WORDS = ("box", "chair", "sit")


def name_rejected(name: str) -> bool:
    """True when the clip name matches the Appendix E exclusion list."""
    text = name.lower().replace("_", " ").replace("-", " ")
    if any(phrase in text for phrase in _PHRASES):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", text))
    return any(word in tokens for word in _WORDS)


def feasibility_reject(
    zmin: torch.Tensor,
    foot_clearance: torch.Tensor,
    pelvis_height: torch.Tensor,
    upright: torch.Tensor,
    speed: torch.Tensor,
    fps: float = 50.0,
) -> str | None:
    """Return a reason, or None when the clip is kept.

    ``zmin`` is the lowest body-link height. ``foot_clearance`` is ``[T, 2]``
    geom-to-floor distance. ``upright`` is true when the pelvis is upright.
    ``speed`` is the root speed.
    """
    frames = max(int(round(0.6 * fps)), 1)
    above = zmin > 0.20
    if _run(above, frames):
        return "elevated support"
    if bool((zmin < -0.12).any()):
        return "ground penetration"
    if float((zmin < -0.08).float().mean()) > 0.80:
        return "ground penetration"
    both_feet = (foot_clearance > 0.02).all(dim=-1)
    still = upright & (pelvis_height > 0.6) & (pelvis_height < 1.1) & (speed < 0.15)
    if _run(both_feet & still, max(int(round(0.5 * fps)), 1)):
        return "floating feet"
    return None


def upright_from_state(state: torch.Tensor) -> torch.Tensor:
    """Upright mask from the root quaternion at the state's own times.

    A missing ``upright`` field is derived here. It is not treated as true.
    """
    gravity_z = projected_gravity(state[:, 3:7])[:, 2]
    return gravity_z < UPRIGHT_GRAVITY_Z


def _resample_mask(mask: torch.Tensor, src_fps: float, dst_fps: float) -> torch.Tensor:
    """Nearest source frame on the same grid as ``resample_series``."""
    src_t, dst_t, _dt = control_time_grid(int(mask.shape[0]), src_fps, dst_fps)
    idx = torch.searchsorted(src_t, dst_t).clamp(max=mask.shape[0] - 1)
    prev = (idx - 1).clamp_min(0)
    choose_prev = (dst_t - src_t[prev]).abs() <= (src_t[idx] - dst_t).abs()
    nearest = torch.where(choose_prev, prev, idx)
    return mask[nearest]


def _run(mask: torch.Tensor, needed: int) -> bool:
    count = 0
    for value in mask.tolist():
        count = count + 1 if value else 0
        if count >= needed:
            return True
    return False


def feasibility_reason(payload: dict, state: torch.Tensor, source: str) -> str | None:
    """Appendix E feasibility gates for one decoded clip, or None if kept.

    ``payload`` holds ``zmin`` and ``foot_clearance`` on the source time
    axis; ``state`` is the 50 Hz decoded clip.
    """
    if "zmin" not in payload or "foot_clearance" not in payload:
        raise ValueError(f"{source} must provide both zmin and foot_clearance")
    source_frames = int(torch.as_tensor(payload["state"]).shape[0])
    src_fps = float(payload["fps"])
    zmin = torch.as_tensor(payload["zmin"], dtype=torch.float32).reshape(-1)
    feet = torch.as_tensor(payload["foot_clearance"], dtype=torch.float32)
    if feet.ndim != 2 or feet.shape[1] != 2 or feet.shape[0] != zmin.shape[0]:
        raise ValueError(
            f"{source} foot_clearance must be [T, 2] and match zmin, "
            f"got {tuple(feet.shape)} and {tuple(zmin.shape)}"
        )
    if zmin.shape[0] != source_frames:
        raise ValueError(
            f"{source} zmin has {zmin.shape[0]} frames but the state has "
            f"{source_frames}. Feasibility arrays use the source time axis."
        )
    if not torch.isfinite(zmin).all() or not torch.isfinite(feet).all():
        raise ValueError(f"{source} feasibility arrays must be finite")
    if abs(src_fps - 30.0) < 1e-3:
        # Same 50 Hz grid as the resampled state. Linear in height.
        zmin = resample_series(zmin.unsqueeze(-1), src_fps, 50.0).squeeze(-1)
        feet = resample_series(feet, src_fps, 50.0)
    elif abs(src_fps - 50.0) >= 1e-3:
        raise ValueError(f"{source} fps {src_fps} cannot be aligned to 50 Hz")
    if zmin.shape[0] != state.shape[0] or feet.shape != (state.shape[0], 2):
        raise ValueError(
            f"{source} feasibility length {zmin.shape[0]} does not match "
            f"the {state.shape[0]}-frame state. A 30 Hz array is not "
            "used as if it were already 50 Hz."
        )
    upright = upright_from_state(state)
    if "upright" in payload:
        given = torch.as_tensor(payload["upright"], dtype=torch.bool).reshape(-1)
        if given.numel() != source_frames:
            raise ValueError(
                f"{source} upright has {given.numel()} frames, "
                f"state has {source_frames}"
            )
        if source_frames != state.shape[0]:
            given = _resample_mask(given, src_fps, 50.0)
        if given.shape != upright.shape or not torch.equal(given, upright):
            raise ValueError(
                f"{source} upright does not match the root orientation. "
                "Omit the field to derive it, or pass that derived mask."
            )
    pelvis = state[:, 2]
    speed = state[:, 36:39].norm(dim=-1)
    return feasibility_reject(zmin, feet, pelvis, upright, speed, fps=50.0)


_PACK_FIELDS = {
    "root_pos_w": 3,
    "root_quat_w": 4,
    "root_lin_vel_w": 3,
    "root_ang_vel_w": 3,
    "joint_pos": 29,
    "joint_vel": 29,
}


def _looks_packed(directory: Path) -> bool:
    """True when the directory is a packed motion library, even if incomplete."""
    return any(
        (directory / name).is_file()
        for name in ("motion_ends.npy", "joint_pos.npy", "joint_names.json")
    )


def _load_field(directory: Path, name: str, rows: int, cols: int) -> torch.Tensor:
    import numpy as np

    path = directory / f"{name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{directory} is a packed library but has no {name}.npy")
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2 or array.shape[1] != cols or array.shape[0] < rows:
        raise ValueError(
            f"{path} has shape {tuple(array.shape)}, expected at least [{rows}, {cols}]"
        )
    copied = np.empty((rows, cols), dtype=np.float32)
    np.copyto(copied, array[:rows])
    return torch.from_numpy(copied)


def _load_packed_bones_library(directory: Path, num_clips: int | None) -> MotionLibrary:
    """Load one packed directory of ``.npy`` fields.

    This is the official layout: one array per field, ``motion_ends.npy`` as
    the exclusive cumsum, and ``joint_names.json``. ``root_ang_vel_w`` is the
    world-frame rate; the library stores the body-frame rate. Name filtering
    already happened when the directory was packed, so startup does not open
    a file per clip. Optional ``q0.npy`` is the PD default pose; without it
    the mjlab G1 keyframe is used. Optional ``motion_names.json`` names the
    clips. The first body-state query writes tracked-body kinematics under
    ``body_kinematics/``; a later process maps that directory.
    """
    import json

    import numpy as np

    from rlinf.envs.sim.g1_mimic.constants import (
        G1_JOINT_NAMES,
        JOINT_POS,
        JOINT_VEL,
        ROOT_ANG_VEL,
        ROOT_LIN_VEL,
        ROOT_POS,
        ROOT_QUAT,
        STATE_DIM,
    )
    from rlinf.envs.sim.g1_mimic.kinematics.features import quat_to_rotmat

    names_path = directory / "joint_names.json"
    ends_path = directory / "motion_ends.npy"
    if not names_path.is_file() or not ends_path.is_file():
        raise FileNotFoundError(
            f"{directory} looks like a packed motion library but is missing "
            "joint_names.json or motion_ends.npy"
        )
    joint_names = json.loads(names_path.read_text())
    if list(joint_names) != list(G1_JOINT_NAMES):
        raise ValueError(
            f"{directory} joint order does not match the G1 29-dof list"
        )
    ends = torch.from_numpy(np.asarray(np.load(ends_path), dtype=np.int64)).reshape(-1)
    total = int(ends[-1]) if ends.numel() else 0
    for name, cols in _PACK_FIELDS.items():
        path = directory / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"{directory} is a packed library but has no {name}.npy")
    if num_clips is not None:
        if int(ends.numel()) < num_clips:
            raise FileNotFoundError(
                f"{directory} has {int(ends.numel())} packed clips, need {num_clips}."
            )
        ends = ends[:num_clips]
        total = int(ends[-1])
    frames = torch.empty(total, STATE_DIM, dtype=torch.float32)
    frames[:, ROOT_POS] = _load_field(directory, "root_pos_w", total, 3)
    quat = _load_field(directory, "root_quat_w", total, 4)
    frames[:, ROOT_QUAT] = quat
    frames[:, JOINT_POS] = _load_field(directory, "joint_pos", total, 29)
    frames[:, ROOT_LIN_VEL] = _load_field(directory, "root_lin_vel_w", total, 3)
    ang_w = _load_field(directory, "root_ang_vel_w", total, 3)
    frames[:, ROOT_ANG_VEL] = torch.einsum("tji,tj->ti", quat_to_rotmat(quat), ang_w)
    frames[:, JOINT_VEL] = _load_field(directory, "joint_vel", total, 29)
    if not torch.isfinite(frames).all():
        raise ValueError(f"{directory} packed motion has a non-finite value")
    quat_norm = frames[:, ROOT_QUAT].norm(dim=-1)
    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-2):
        raise ValueError(f"{directory} root quaternion is not normalized")
    q0_path = directory / "q0.npy"
    if q0_path.is_file():
        q0 = torch.from_numpy(np.asarray(np.load(q0_path), dtype=np.float32)).reshape(-1)
        if q0.shape != (29,):
            raise ValueError(f"{q0_path} has shape {tuple(q0.shape)}, expected (29,)")
    else:
        from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import asset_default_state

        q0 = asset_default_state()[JOINT_POS].clone()
    clip_names = None
    names_file = directory / "motion_names.json"
    if names_file.is_file():
        clip_names = [str(name) for name in json.loads(names_file.read_text())]
        if num_clips is not None:
            clip_names = clip_names[:num_clips]
    library = MotionLibrary.from_block(
        frames,
        ends,
        q0=q0,
        names=clip_names,
        kin_cache_dir=directory / "body_kinematics",
    )
    library.filter_counts = {
        "files_read": library.num_clips,
        "files_in_dir": library.num_clips,
        "rejected_name": 0,
        "rejected_feasibility": 0,
        "kept": library.num_clips,
        "packed": True,
    }
    get_logger().info(
        f"BONES-SEED packed clips from {directory}: {library.filter_counts}; "
        f"body kinematics cache {library._kin_cache_dir}"
    )
    return library


def load_bones_library(path: str, num_clips: int | None) -> MotionLibrary:
    """Load ``clip_*.pt`` files, or one packed directory of ``.npy`` fields.

    ``None`` keeps every clip that passes the filters. A count stops after
    that many kept clips and raises if the filters leave fewer.

    Each ``clip_*.pt`` uses the G1 payload (fps, 29 joint names, q0, state)
    plus ``name``. Optional ``zmin`` ``[T]`` and ``foot_clearance`` ``[T, 2]``
    turn on the feasibility filter. Missing arrays skip that filter and the
    caller must not treat the load as a full Appendix E pass.

    A directory with ``motion_ends.npy`` and ``joint_pos.npy`` is the packed
    layout instead. Filtering is not repeated there.

    The library stays on CPU; ``state_at`` copies only the gathered rows to
    the querying device.
    """
    directory = Path(path)
    if directory.is_dir() and _looks_packed(directory):
        return _load_packed_bones_library(directory, num_clips)
    files = sorted(directory.glob("clip_*.pt")) if directory.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"No G1-retargeted BONES-SEED clips at {path}. "
            "Refusing to substitute human motion or synthetic clips."
        )
    kept_state = []
    kept_names: list[str] = []
    q0 = None
    skipped = {"name": 0, "feasibility": 0}
    for file in files:
        payload = torch.load(file, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "name" not in payload:
            raise ValueError(
                f"{file} has no clip name. BONES-SEED filtering needs one."
            )
        name = str(payload["name"])
        if name_rejected(name):
            skipped["name"] += 1
            continue
        state, clip_q0 = decode_motion_payload(payload, str(file))
        if "zmin" in payload or "foot_clearance" in payload:
            if feasibility_reason(payload, state, str(file)) is not None:
                skipped["feasibility"] += 1
                continue
        if q0 is None:
            q0 = clip_q0
        elif not torch.allclose(q0, clip_q0, atol=1e-5):
            raise ValueError(f"{file} default pose does not match the other clips")
        kept_state.append(state)
        kept_names.append(name)
        if num_clips is not None and len(kept_state) >= num_clips:
            break
    if not kept_state:
        raise RuntimeError(
            f"BONES-SEED filter removed every clip in {path} "
            f"(name={skipped['name']}, feasibility={skipped['feasibility']})."
        )
    if num_clips is not None and len(kept_state) < num_clips:
        raise FileNotFoundError(
            f"{directory} kept {len(kept_state)} clips after filtering, need {num_clips}."
        )
    library = MotionLibrary(kept_state, q0=q0)
    library.clip_names = kept_names
    library.filter_counts = {
        "files_read": len(kept_state) + skipped["name"] + skipped["feasibility"],
        "files_in_dir": len(files),
        "rejected_name": skipped["name"],
        "rejected_feasibility": skipped["feasibility"],
        "kept": len(kept_state),
    }
    get_logger().info(f"BONES-SEED clips from {directory}: {library.filter_counts}")
    return library
