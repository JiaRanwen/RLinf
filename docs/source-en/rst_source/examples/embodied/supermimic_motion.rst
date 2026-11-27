G1 Motion Data Preparation
==========================

Prepare reference motion clips for the :doc:`G1 Motion Tracking Environment <g1_mimic>` and :doc:`SuperMimic <supermimic>`, including offline LAFAN1 conversion, the experimental BONES-SEED path, and custom ``clip_*.pt`` files.

**Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the ``.venv-supermimic`` environment from :doc:`supermimic`.

Setup and Run
-------------

Activate ``.venv-supermimic`` and run all commands from the RLinf repository root.

.. tab-set::

   .. tab-item:: Offline LAFAN1 Conversion

      If the training machine cannot reach Hugging Face, convert LAFAN1 on an internet-connected machine that has an RLinf checkout and Python with ``torch``, ``numpy``, and ``omegaconf``. Because conversion checks against mjlab's G1 default joint pose, export that pose from the training machine first:

      **1. Export the default joint pose on the training machine**

      .. code-block:: bash

         python toolkits/supermimic/prepare_lafan1.py --export-q0 ./g1_q0.pt

      **2. Copy ``g1_q0.pt`` and convert the 40 clips on the internet-connected machine**

      .. code-block:: bash

         python toolkits/supermimic/prepare_lafan1.py --out ./data/lafan1_g1 \
             --q0-file ./g1_q0.pt --accept-license

      **3. Copy ``./data/lafan1_g1`` (including ``manifest.json``) back to the training machine**

      Point ``SUPERMIMIC_MOTION_PATH`` to ``./data/lafan1_g1`` and launch training as shown in :doc:`supermimic`.

   .. tab-item:: Custom Clips

      Save each custom clip in the ``clip_*.pt`` format below inside one folder (loaded in file-name order), point ``SUPERMIMIC_MOTION_PATH`` to it, and pass ``env.train.num_clips=null`` (or ``env.eval.num_clips=null`` for evaluation) to load every clip:

      .. code-block:: bash

         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/my_clips
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord \
             env.train.num_clips=null

      To evaluate more than 40 clips, adjust the evaluation capacity settings in :doc:`../../evaluations/guides/supermimic`. If the directory includes a ``manifest.json``, the loader verifies every clip's SHA-256 and rejects unlisted files; without ``manifest.json``, it validates tensor shapes and values only.

   .. tab-item:: BONES-SEED (Experimental)

      The public G1 retargeting is the Hugging Face dataset `bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_. Accept the license on the dataset page, then download ``g1.tar.gz`` and ``LICENSE.md``:

      .. code-block:: bash

         hf download bones-studio/seed g1.tar.gz LICENSE.md --repo-type dataset --local-dir ./data/bones-seed
         tar -xzf ./data/bones-seed/g1.tar.gz -C ./data/bones-seed

      If the dataset asks you to sign in, run ``hf auth login`` first. The CSVs are under ``./data/bones-seed/g1/csv``.

      Conversion resamples the 120 Hz CSVs to 50 Hz ``clip_*.pt`` files and keeps every clip:

      .. code-block:: bash

         python toolkits/supermimic/convert_bones_seed.py \
             --input ./data/bones-seed/g1/csv \
             --output ./data/bones_g1 \
             --jobs 16

      A successful run ends with ``[bones-seed] done: 142220 converted, 0 skipped, 0 failed``.

      Loading keeps clips whose names match the terms named in Appendix E: 135,798 of 142,220. The paper's name filter keeps 129,667 and does not publish the full term list. A packed directory, one ``.npy`` file per field plus ``motion_ends.npy``, avoids opening a file per clip. ``root_ang_vel_w`` is stored in the world frame and converted to the body frame on load.

      Train GenCoord on that library for 80,000 updates. The log should report ``kept`` as 135798.

      .. code-block:: bash

         export MUJOCO_GL=disabled
         export SUPERMIMIC_MOTION_PATH=./data/bones_g1
         bash examples/embodiment/run_embodiment.sh supermimic_lafan1_gencoord \
             env.train.motion_format=bones_seed env.eval.motion_format=bones_seed \
             env.train.num_clips=null env.eval.num_clips=null \
             runner.max_epochs=80000

      Full-library evaluation has to cover all 135,798 kept clips. ``env.eval.total_num_envs`` times ``env.eval.rollout_epoch`` must be at least that count; 16,384 and 9 do, with the envs split across four GPUs. Set ``max_steps_per_rollout_epoch`` to the longest kept clip in control steps (frame count minus 1).

This training run
~~~~~~~~~~~~~~~~~

The name filter keeps 135,798 clips. One GenCoord run on 4 GPUs in bf16 trained them for 80,000 updates, saved the 20,000 and 80,000 checkpoints, and evaluated both on the full library.

.. list-table::
   :header-rows: 1
   :widths: 14 22 18 24 22

   * - Updates
     - ``normalized_return``
     - Falls
     - ``body_pos_err`` (m)
     - ``torso_ori_err``
   * - 80000
     - 0.937
     - 421 (0.310%)
     - 0.021
     - 0.079
   * - 20000
     - 0.927
     - 830 (0.611%)
     - 0.023
     - —

On the 80,000 checkpoint, 0.997 of the 135,798 clips play to the end. A clip lasts 7.34 s on average.

Numbers recorded during training. The per-step tracking reward tops out at 0.1.

.. list-table::
   :header-rows: 1
   :widths: 16 18 20 22 24

   * - Updates
     - Step reward
     - Fall fraction
     - Actor return
     - ``state_loss``
   * - 20000
     - 0.085
     - 0.0028
     - 13.84
     - 0.0091
   * - 40000
     - 0.086
     - 0.0032
     - 13.91
     - 0.0072
   * - 80000
     - 0.087
     - 0.0033
     - 14.04
     - 0.0077

Clip File Format
----------------

Each ``clip_*.pt`` file is a Python dictionary saved with ``torch.save``:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Contents
   * - ``fps``
     - ``50`` or ``30``. A 30 Hz clip is resampled to the 50 Hz control grid and its velocities are recomputed by central difference. A 50 Hz clip is loaded as stored, so **its velocity columns must be accurate**. Other rates are rejected.
   * - ``joint_names``
     - List of 29 joint names in the exact mjlab Unitree G1 order (left leg, right leg, waist, left arm, right arm).
   * - ``q0``
     - 29-dim default joint pose in radians, matching mjlab's standing default pose.
   * - ``state``
     - Float tensor of shape ``[T, 71]`` with the column layout below, finite values only, and unit-norm root quaternions.

Column layout of ``state`` (``[T, 71]``):

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Columns
     - Contents
     - Unit
   * - ``0:3``
     - Root (pelvis) position, world frame
     - m
   * - ``3:7``
     - Root orientation quaternion (``wxyz``)
     - --
   * - ``7:36``
     - Joint positions, in ``joint_names`` order
     - rad
   * - ``36:39``
     - Root linear velocity, world frame
     - m/s
   * - ``39:42``
     - Root angular velocity, root body frame
     - rad/s
   * - ``42:71``
     - Joint velocities
     - rad/s

Export ``g1_q0.pt`` once with ``python toolkits/supermimic/prepare_lafan1.py --export-q0 ./g1_q0.pt``, then save each custom clip dictionary with ``torch.save`` (for example ``clip_000.pt``). The loader validates every clip's keys, joint order, ``q0`` match, quaternion norms, and finite values on startup.
