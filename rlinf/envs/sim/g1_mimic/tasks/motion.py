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

"""Reference clips, goal windows and the goal vectors built from them."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from omegaconf import DictConfig

from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    CONTROL_FPS,
    G1_JOINT_NAMES,
    GOAL_FRAMES,
    GOAL_STRIDE,
    JOINT_POS,
    JOINT_VEL,
    ROOT_ANG_VEL,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
    STATE_DIM,
)
from rlinf.envs.sim.g1_mimic.kinematics.features import (
    quat_mul,
    quat_to_rotmat,
    rot_apply_t,
    rot_mul_t,
    rotmat_to_6d,
    rotvec_from_quat,
)


# Bump when the on-disk body-kinematics layout changes. The file digest also
# hashes one probe pose, so a change to the kinematic tree misses this file
# even when the version stays put.
_BODY_KIN_VERSION = 1
_BODY_KIN_NAMES = ("pos", "rot", "lin", "ang")


def _log_motion(message: str) -> None:
    from rlinf.utils.logging import get_logger

    get_logger().info(message)


def _digest_array(digest: hashlib._Hash, array: object) -> None:
    import numpy as np

    values = np.ascontiguousarray(array)
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())


@contextmanager
def _exclusive_lock(path: Path):
    """Block until this process holds an exclusive lock on ``path``."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def assemble_goals(
    content: torch.Tensor,
    root_pos: torch.Tensor,
    root_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-model goal and actor goal of one root pose.

    ``content`` is the joint part of the window ``[..., 580]``; ``target_pos``
    and ``target_rot`` are the window's reference root ``[..., 10, 3]`` and
    ``[..., 10, 3, 3]``. Both goals express the targets in the root frame;
    the actor goal leaves out the relative positions.
    """
    frame = root_rot.unsqueeze(-3)
    rel_pos = rot_apply_t(frame, target_pos - root_pos.unsqueeze(-2))
    rel_rot = rotmat_to_6d(rot_mul_t(frame, target_rot)).flatten(-2)
    world_model_goal = torch.cat((content, rel_pos.flatten(-2), rel_rot), dim=-1)
    actor_goal = torch.cat((content, rel_rot), dim=-1)
    return world_model_goal, actor_goal


def root_goals(
    root: torch.Tensor,
    content: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-model and actor goals of the goal window seen from ``root``."""
    return assemble_goals(
        content, root[..., :3], quat_to_rotmat(root[..., 3:7]), target_pos, target_rot
    )


class MotionLibrary:
    """All clips in one contiguous ``[sum_T, 71]`` block.

    Clip ``i`` is rows ``offsets[i] .. ends[i] - 1``. The simulator addresses
    a frame by clip id and time; the replay buffer, the reset sampler and
    the world model address it by its row in the block (a global frame).
    Indexing stays on the storage device and only the gathered rows move to
    the query device, so a CPU library is not copied onto the GPU.
    """

    def __init__(self, clips: list[torch.Tensor], q0: torch.Tensor | None = None):
        if not clips:
            raise ValueError("MotionLibrary needs at least one clip, got none")
        lengths = [int(c.shape[0]) for c in clips]
        if min(lengths) < 1:
            raise ValueError(
                "every reference clip needs at least one frame; clip "
                f"{lengths.index(min(lengths))} has {min(lengths)}"
            )
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.ends = torch.cumsum(self.lengths, dim=0)
        self.offsets = self.ends - self.lengths
        self.frames = torch.cat([c.float() for c in clips], dim=0)
        self._index_cache: dict[torch.device, tuple[torch.Tensor, ...]] = {}
        # Body position, rotation, linear and angular velocity of every frame,
        # on the frames' storage device. The clips do not change, so training
        # gathers these instead of walking the kinematic tree again.
        self._kin_cache: tuple[torch.Tensor, ...] | None = None
        self._kin_cache_dir: Path | None = None
        self.clip_names = [f"clip_{i:03d}" for i in range(len(clips))]
        self.q0 = (
            q0.float()
            if q0 is not None
            else torch.zeros(ACTION_DIM, dtype=torch.float32)
        )

    @classmethod
    def from_block(
        cls,
        frames: torch.Tensor,
        ends: torch.Tensor,
        q0: torch.Tensor | None = None,
        names: list[str] | None = None,
        kin_cache_dir: str | Path | None = None,
    ) -> MotionLibrary:
        """Build a library from one ``[sum_T, 71]`` block and exclusive ends.

        ``ends[i]`` is the first row of the next clip. This skips concatenating
        one tensor per clip, which is the packed-directory load.
        ``kin_cache_dir`` stores the full-library body kinematics after the
        first query. A later process maps those files instead of recomputing.
        """
        frames = frames.float().contiguous()
        ends = ends.to(dtype=torch.long).reshape(-1).contiguous()
        if frames.ndim != 2 or frames.shape[-1] != STATE_DIM:
            raise ValueError(
                f"packed frames have shape {tuple(frames.shape)}, expected [T, {STATE_DIM}]"
            )
        if ends.numel() == 0 or int(ends[-1]) != frames.shape[0]:
            raise ValueError(
                f"motion ends {tuple(ends.shape)} do not cover {frames.shape[0]} frames"
            )
        if int(ends[0]) < 1 or bool((ends[1:] <= ends[:-1]).any()):
            raise ValueError("motion ends must be positive and strictly increasing")
        if names is not None and len(names) != int(ends.numel()):
            raise ValueError(
                f"{len(names)} clip names for {int(ends.numel())} packed clips"
            )
        library = cls.__new__(cls)
        library.ends = ends
        library.lengths = ends.clone()
        library.lengths[1:] = ends[1:] - ends[:-1]
        library.offsets = ends - library.lengths
        library.frames = frames
        library._index_cache = {}
        library._kin_cache = None
        library._kin_cache_dir = (
            None if kin_cache_dir is None else Path(kin_cache_dir)
        )
        library.clip_names = (
            list(names)
            if names is not None
            else [f"clip_{i:03d}" for i in range(int(ends.numel()))]
        )
        library.q0 = (
            q0.float()
            if q0 is not None
            else torch.zeros(ACTION_DIM, dtype=torch.float32)
        )
        return library

    @property
    def num_clips(self) -> int:
        """Number of clips."""
        return int(self.lengths.shape[0])

    @property
    def num_frames(self) -> int:
        """Total frames over all clips."""
        return int(self.frames.shape[0])

    def _index(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if device not in self._index_cache:
            self._index_cache[device] = (
                self.offsets.to(device=device),
                self.lengths.to(device=device),
                self.ends.to(device=device),
            )
        return self._index_cache[device]

    def _store(self, frame: torch.Tensor) -> torch.Tensor:
        return frame.to(device=self.frames.device, dtype=torch.long)

    def clip_length(self, motion_id: torch.Tensor) -> torch.Tensor:
        """Frame count of each clip in ``motion_id``."""
        return self._index(motion_id.device)[1][motion_id]

    def global_frame(self, motion_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Row of frame ``time`` of clip ``motion_id``, held at the clip's ends."""
        offsets, lengths, _ = self._index(motion_id.device)
        mid = motion_id.long()
        index = torch.minimum(time.long().clamp_min(0), lengths[mid] - 1)
        return offsets[mid] + index

    def locate(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Clip id and time of each global frame."""
        offsets, _, ends = self._index(frame.device)
        clip = torch.searchsorted(ends, frame.long(), right=True)
        return clip, frame.long() - offsets[clip]

    def clip_end(self, frame: torch.Tensor) -> torch.Tensor:
        """Exclusive end row of the clip that holds each global frame."""
        ends = self._index(frame.device)[2]
        clip = torch.searchsorted(ends, frame.long(), right=True)
        return ends[clip.clamp(max=self.num_clips - 1)]

    def rows(self, frame: torch.Tensor) -> torch.Tensor:
        """71-d reference states at global frames."""
        return self.frames[self._store(frame)].to(device=frame.device)

    def _kin_cache_identity(self) -> str:
        """Digest of these frames and of one probe through the current tree."""
        from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import (
            TRACKED_BODY_NAMES,
            g1_kinematics,
        )

        digest = hashlib.sha256()
        digest.update(f"body-kinematics-v{_BODY_KIN_VERSION}".encode())
        digest.update("\0".join(TRACKED_BODY_NAMES).encode())
        probe = torch.zeros(1, STATE_DIM)
        probe[:, ROOT_QUAT] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        probe[:, ROOT_POS] = torch.tensor([0.1, -0.2, 0.8])
        probe[:, JOINT_POS] = 0.15
        probe[:, ROOT_LIN_VEL] = torch.tensor([0.3, -0.1, 0.05])
        probe[:, ROOT_ANG_VEL] = torch.tensor([0.2, -0.3, 0.4])
        probe[:, JOINT_VEL] = -0.05
        with torch.no_grad():
            probed = g1_kinematics().forward(probe, compile=False)
        for item in probed:
            _digest_array(digest, item.detach().cpu().contiguous().numpy())
        ends = self.ends.detach().cpu().contiguous()
        _digest_array(digest, ends.numpy())
        frames = self.frames.detach().cpu()
        sample = torch.cat((frames[::1024], frames[-1:]), dim=0)
        _digest_array(digest, sample.contiguous().numpy())
        return digest.hexdigest()

    def _compute_kinematics(self) -> tuple[torch.Tensor, ...]:
        from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import g1_kinematics

        kin = g1_kinematics()
        total = self.num_frames
        outputs: tuple[torch.Tensor, ...] | None = None
        logged = time.monotonic()
        with torch.no_grad():
            for start in range(0, total, 4096):
                batch = self.frames[start : start + 4096]
                parts = tuple(
                    item.detach() for item in kin.forward(batch, compile=False)
                )
                if outputs is None:
                    outputs = tuple(
                        torch.empty(
                            total,
                            *item.shape[1:],
                            dtype=item.dtype,
                            device=item.device,
                        )
                        for item in parts
                    )
                stop = start + parts[0].shape[0]
                for dest, item in zip(outputs, parts, strict=True):
                    dest[start:stop] = item
                now = time.monotonic()
                if now - logged >= 60.0:
                    _log_motion(f"body kinematics computed {stop}/{total} frames")
                    logged = now
        if outputs is None:
            raise RuntimeError("body kinematics received no frames")
        return outputs

    def _read_kin_cache(
        self, dest: Path, identity: str
    ) -> tuple[torch.Tensor, ...] | None:
        import numpy as np

        if not (dest / "ready").is_file() or not (dest / "meta.json").is_file():
            return None
        try:
            meta = json.loads((dest / "meta.json").read_text())
        except json.JSONDecodeError:
            return None
        shapes = meta.get("shapes")
        if (
            meta.get("version") != _BODY_KIN_VERSION
            or meta.get("digest") != identity
            or int(meta.get("num_frames", -1)) != self.num_frames
            or meta.get("names") != list(_BODY_KIN_NAMES)
            or not isinstance(shapes, list)
            or len(shapes) != len(_BODY_KIN_NAMES)
        ):
            return None
        tensors = []
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The given NumPy array is not writable",
                )
                for name, shape in zip(_BODY_KIN_NAMES, shapes, strict=True):
                    path = dest / f"{name}.npy"
                    array = np.load(path, mmap_mode="c")
                    if array.dtype != np.float32 or list(array.shape) != list(shape):
                        return None
                    if int(array.shape[0]) != self.num_frames:
                        return None
                    if path.stat().st_size < array.nbytes:
                        return None
                    tensors.append(torch.from_numpy(array))
        except (OSError, ValueError, EOFError):
            return None
        return tuple(tensors)

    def _write_kin_cache(
        self, dest: Path, identity: str, tensors: tuple[torch.Tensor, ...]
    ) -> None:
        import numpy as np

        if len(tensors) != len(_BODY_KIN_NAMES):
            raise RuntimeError(
                f"body kinematics returned {len(tensors)} tensors, "
                f"expected {len(_BODY_KIN_NAMES)}"
            )
        dest.mkdir(parents=True, exist_ok=True)
        os.chmod(dest, 0o755)
        ready = dest / "ready"
        if ready.exists():
            ready.unlink()
        shapes = []
        for name, tensor in zip(_BODY_KIN_NAMES, tensors, strict=True):
            array = tensor.detach().contiguous().numpy()
            if array.dtype != np.float32:
                array = np.asarray(array, dtype=np.float32)
            partial = dest / f"{name}.npy.partial"
            with partial.open("wb") as handle:
                np.save(handle, array)
            final = dest / f"{name}.npy"
            os.replace(partial, final)
            os.chmod(final, 0o644)
            shapes.append(list(array.shape))
            _log_motion(f"wrote {final} shape {tuple(array.shape)}")
        meta = {
            "version": _BODY_KIN_VERSION,
            "digest": identity,
            "num_frames": self.num_frames,
            "dtype": "float32",
            "names": list(_BODY_KIN_NAMES),
            "shapes": shapes,
        }
        meta_partial = dest / "meta.json.partial"
        meta_partial.write_text(json.dumps(meta))
        os.replace(meta_partial, dest / "meta.json")
        os.chmod(dest / "meta.json", 0o644)
        ready_partial = dest / "ready.partial"
        ready_partial.write_text("ok\n")
        os.replace(ready_partial, ready)
        os.chmod(ready, 0o644)

    def _kinematics(self) -> tuple[torch.Tensor, ...]:
        """Tracked-body position, rotation, linear and angular velocity.

        The four tensors cover every frame. With a cache directory set, one
        process writes them and the others map those files. The directory
        name is a digest of the frames and of one probe pose, so a different
        clip or a changed kinematic tree does not reuse the file.
        """
        if self._kin_cache is not None:
            return self._kin_cache
        cache_dir = self._kin_cache_dir
        if cache_dir is None or self.frames.device.type != "cpu":
            self._kin_cache = self._compute_kinematics()
            return self._kin_cache
        _log_motion(
            f"body kinematics: {self.num_frames} frames, cache directory {cache_dir}"
        )
        with _exclusive_lock(cache_dir / ".lock"):
            identity = self._kin_cache_identity()
            dest = cache_dir / identity
            cached = self._read_kin_cache(dest, identity)
            if cached is None:
                started = time.monotonic()
                _log_motion(
                    "computing body kinematics for "
                    f"{self.num_frames} frames into {dest}"
                )
                computed = self._compute_kinematics()
                self._write_kin_cache(dest, identity, computed)
                del computed
                import gc

                gc.collect()
                cached = self._read_kin_cache(dest, identity)
                if cached is None:
                    raise RuntimeError(
                        f"body kinematics cache at {dest} could not be read back"
                    )
                elapsed = time.monotonic() - started
                _log_motion(
                    f"wrote body kinematics cache {dest} in {elapsed:.0f}s"
                )
            else:
                _log_motion(f"loaded body kinematics cache {dest}")
            self._kin_cache = cached
        return self._kin_cache

    def body_state(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tracked-body world position, rotation, linear and angular velocity."""
        rows = self._store(frame)
        return tuple(item[rows].to(device=frame.device) for item in self._kinematics())

    def state_at(self, motion_id: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """71-d reference states by clip id and time."""
        return self.rows(self.global_frame(motion_id, time))

    def goal_window(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Joint content ``[..., 580]`` and root targets of the goal window.

        The window is ten frames ``GOAL_STRIDE`` apart from ``frame``, held at
        the clip's last frame. Returns the content, the target root positions
        ``[..., 10, 3]`` and rotations ``[..., 10, 3, 3]``.
        """
        steps = GOAL_STRIDE * torch.arange(GOAL_FRAMES, device=frame.device)
        window = torch.minimum(
            frame.long().unsqueeze(-1) + steps,
            (self.clip_end(frame) - 1).unsqueeze(-1),
        )
        states = self.rows(window)
        content = torch.cat(
            (states[..., JOINT_POS], states[..., JOINT_VEL]), dim=-1
        ).flatten(-2)
        return content, states[..., :3], quat_to_rotmat(states[..., ROOT_QUAT])


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def _slerp(q0: torch.Tensor, q1: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    q0 = torch.nn.functional.normalize(q0, dim=-1)
    q1 = torch.nn.functional.normalize(q1, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(-1.0, 1.0)
    # Fall back to normalized lerp when the quaternions are nearly identical.
    close = (dot > 0.9995).squeeze(-1)
    u = u.unsqueeze(-1)
    lerped = torch.nn.functional.normalize((1.0 - u) * q0 + u * q1, dim=-1)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega).clamp_min(1e-8)
    slerped = (
        torch.sin((1.0 - u) * omega) * q0 + torch.sin(u * omega) * q1
    ) / sin_omega
    return torch.where(close.unsqueeze(-1), lerped, slerped)


def control_time_grid(
    num_frames: int, src_fps: float, dst_fps: float
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Source sample times and the destination times on ``1 / dst_fps``.

    Destination times are ``0, dt, 2 dt, ...`` while ``t`` is still inside
    the source duration ``(num_frames - 1) / src_fps``. A tail shorter than
    one control step is dropped. It is not stretched onto the source
    endpoint, so every finite difference uses the same ``dt``.
    """
    if num_frames < 2:
        raise ValueError(
            f"a reference clip needs at least two frames to resample, got {num_frames}"
        )
    dt = 1.0 / float(dst_fps)
    # i / dst_fps <= (num_frames - 1) / src_fps, in exact rational form.
    steps = int(math.floor((num_frames - 1) * float(dst_fps) / float(src_fps) + 1e-8))
    if steps < 1:
        raise ValueError(
            f"a clip of {num_frames} frames at {src_fps} fps is shorter than one "
            f"{dst_fps} Hz control step"
        )
    src_t = torch.arange(num_frames, dtype=torch.float64) / float(src_fps)
    dst_t = torch.arange(steps + 1, dtype=torch.float64) * dt
    return src_t, dst_t, dt


def _interp_weights(
    src_t: torch.Tensor, dst_t: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.searchsorted(src_t, dst_t, right=True).clamp(1, count - 1)
    t0 = src_t[idx - 1]
    t1 = src_t[idx]
    u = (dst_t - t0) / (t1 - t0).clamp_min(1e-12)
    return idx, u


def resample_series(
    values: torch.Tensor, src_fps: float, dst_fps: float = CONTROL_FPS
) -> torch.Tensor:
    """Linear resample of a per-frame array onto the control-time grid."""
    values = torch.as_tensor(values, dtype=torch.float32)
    count = int(values.shape[0])
    src_t, dst_t, _dt = control_time_grid(count, src_fps, dst_fps)
    idx, u = _interp_weights(src_t, dst_t, count)
    shape = [u.shape[0]] + [1] * (values.ndim - 1)
    u = u.to(dtype=values.dtype).reshape(shape)
    return (1.0 - u) * values[idx - 1] + u * values[idx]


def _central_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    """Rate at each sample: central inside, one-sided at the two ends."""
    rate = torch.empty_like(values)
    rate[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    rate[0] = (values[1] - values[0]) / dt
    rate[-1] = (values[-1] - values[-2]) / dt
    return rate


def _root_angular_velocity(quat: torch.Tensor, dt: float) -> torch.Tensor:
    """Root-frame angular velocity from a world-frame central SO(3) difference."""
    world = torch.empty(quat.shape[0], 3, dtype=quat.dtype)
    world[1:-1] = rotvec_from_quat(quat_mul(quat[2:], _quat_inv(quat[:-2]))) / (
        2.0 * dt
    )
    world[0] = rotvec_from_quat(quat_mul(quat[1], _quat_inv(quat[0]))) / dt
    world[-1] = rotvec_from_quat(quat_mul(quat[-1], _quat_inv(quat[-2]))) / dt
    return torch.einsum("tji,tj->ti", quat_to_rotmat(quat), world)


def resample_states(
    state: torch.Tensor, src_fps: float, dst_fps: float = CONTROL_FPS
) -> torch.Tensor:
    """Linear positions, slerp quaternions, velocities on the control grid.

    Root position, joint position, and translation use linear interpolation.
    Root orientation uses slerp, so the result stays unit length and does
    not flip hemisphere between adjacent samples. Velocities are central
    differences on the same grid, so frame ``i`` holds the rate at time
    ``i * dt`` rather than half a step earlier; the two end frames use the
    one-sided difference. Angular velocity is expressed in the root frame.
    """
    count = int(state.shape[0])
    src_t, dst_t, dt = control_time_grid(count, src_fps, dst_fps)
    idx, u = _interp_weights(src_t, dst_t, count)
    u = u.to(dtype=state.dtype)
    u_col = u.unsqueeze(-1)
    out = torch.empty(dst_t.shape[0], STATE_DIM, dtype=state.dtype)
    out[:, :3] = (1.0 - u_col) * state[idx - 1, :3] + u_col * state[idx, :3]
    out[:, JOINT_POS] = (1.0 - u_col) * state[idx - 1, JOINT_POS] + u_col * state[
        idx, JOINT_POS
    ]
    out[:, ROOT_QUAT] = _slerp(state[idx - 1, ROOT_QUAT], state[idx, ROOT_QUAT], u)
    out[:, ROOT_LIN_VEL] = _central_difference(out[:, :3], dt)
    out[:, JOINT_VEL] = _central_difference(out[:, JOINT_POS], dt)
    out[:, ROOT_ANG_VEL] = _root_angular_velocity(out[:, ROOT_QUAT], dt)
    return out


def decode_motion_payload(
    payload: object, source: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check one G1 reference clip and return ``(state [T, 71], q0 [29])``.

    A bare tensor is rejected: it does not say which joint is which, or
    whether the frames are 50 Hz. ``fps`` must be 50, or 30 so the clip can
    be resampled onto the control rate. ``joint_names`` must equal
    ``G1_JOINT_NAMES``. ``q0`` is the default pose the PD offset uses.
    """
    if torch.is_tensor(payload):
        raise ValueError(
            f"{source} is a bare state tensor. A G1 reference needs fps, "
            f"joint_names ({ACTION_DIM} names), q0, and state."
        )
    if not isinstance(payload, dict):
        raise ValueError(
            f"{source} must be a dict with fps, joint_names, q0, and state"
        )
    names = [str(name) for name in payload.get("joint_names", ())]
    if names != list(G1_JOINT_NAMES):
        raise ValueError(
            f"{source} joint order does not match the G1 29-dof list. "
            f"First mismatch or length {len(names)} vs {ACTION_DIM}."
        )
    if "fps" not in payload or "q0" not in payload or "state" not in payload:
        raise ValueError(f"{source} is missing fps, q0, or state")
    fps = float(payload["fps"])
    q0 = torch.as_tensor(payload["q0"], dtype=torch.float32).reshape(-1)
    state = torch.as_tensor(payload["state"], dtype=torch.float32)
    if q0.shape != (ACTION_DIM,):
        raise ValueError(
            f"{source} q0 has shape {tuple(q0.shape)}, expected ({ACTION_DIM},)"
        )
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(
            f"{source} state has shape {tuple(state.shape)}, expected [T, {STATE_DIM}]"
        )
    if not torch.isfinite(state).all() or not torch.isfinite(q0).all():
        raise ValueError(f"{source} has non-finite state or q0")
    quat_norm = state[:, ROOT_QUAT].norm(dim=-1)
    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-2):
        raise ValueError(f"{source} root quaternion is not normalized")
    if abs(fps - CONTROL_FPS) < 1e-3:
        decoded = state
    elif abs(fps - 30.0) < 1e-3:
        decoded = resample_states(state, 30.0, CONTROL_FPS)
    else:
        raise ValueError(
            f"{source} fps is {fps}. Control is {CONTROL_FPS} Hz. "
            "Only a 30 Hz retarget source is resampled; other rates are rejected."
        )
    return decoded, q0


def check_manifest(directory: Path, files: list[Path]) -> list[str]:
    """Verify ``files`` against ``directory/manifest.json`` if there is one.

    Returns each clip's source name, or the file name without a manifest.
    A listed file with another SHA-256, or a file the manifest does not
    list, raises: the clips are not the ones that were prepared.
    """
    path = directory / "manifest.json"
    if not path.is_file():
        return [file.name for file in files]
    manifest = json.loads(path.read_text())
    listed = {clip["file"]: clip for clip in manifest.get("clips", [])}
    names = []
    for file in files:
        entry = listed.get(file.name)
        if entry is None:
            raise ValueError(f"{file} is not listed in {path}")
        digest = hashlib.sha256(file.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"{file} does not match its SHA-256 in {path}")
        names.append(str(entry.get("source", file.name)))
    return names


def load_motion_library(
    path: str | None, num_clips: int | None, device: torch.device | str
) -> MotionLibrary:
    """Load verified ``clip_*.pt`` payloads.

    ``num_clips`` takes the first files in name order; ``None`` takes all of
    them. A missing, short, or unchecked directory raises.
    """
    directory = Path(path) if path else None
    files = (
        sorted(directory.glob("clip_*.pt")) if directory and directory.is_dir() else []
    )
    if files:
        if num_clips is not None and len(files) < num_clips:
            raise FileNotFoundError(
                f"{directory} has {len(files)} clip_*.pt files, need {num_clips}."
            )
        selected = files if num_clips is None else files[:num_clips]
        sources = check_manifest(directory, selected)
        clips = []
        q0 = None
        for file in selected:
            payload = torch.load(file, map_location="cpu", weights_only=True)
            state, clip_q0 = decode_motion_payload(payload, str(file))
            if q0 is None:
                q0 = clip_q0
            elif not torch.allclose(q0, clip_q0, atol=1e-5):
                raise ValueError(f"{file} default pose does not match the other clips")
            clips.append(state.to(device))
        library = MotionLibrary(clips, q0=q0)
        library.clip_names = sources
        return library
    raise FileNotFoundError(
        "No verified G1 reference clips at motion_path. Each clip_*.pt must "
        "contain fps, the 29 G1 joint names, the default pose q0, and state "
        "[T, 71]."
    )


def build_motion_library(env_cfg: DictConfig, device: torch.device) -> MotionLibrary:
    """Load the motion library configured by ``env_cfg``.

    ``lafan1`` reads manifest-checked clips. ``bones_seed`` is the experimental
    Appendix E loader: it filters by clip name, keeps the library on CPU, and
    does not read ``manifest.json``.
    """
    motion_format = str(env_cfg.get("motion_format", "lafan1"))
    num_clips = env_cfg.get("num_clips", None)
    num_clips = None if num_clips is None else int(num_clips)
    if motion_format == "bones_seed":
        # Lazy import: the filter module imports this file's payload decoder.
        from rlinf.envs.sim.g1_mimic.bones import load_bones_library

        return load_bones_library(str(env_cfg.get("motion_path", "")), num_clips)
    if motion_format == "lafan1":
        # motion_on_cpu keeps the frames in host memory; queries gather rows
        # there and copy only those rows to the simulator device.
        store = "cpu" if bool(env_cfg.get("motion_on_cpu", False)) else device
        return load_motion_library(env_cfg.get("motion_path", None), num_clips, store)
    raise ValueError(f"unknown env motion_format {motion_format!r}")
