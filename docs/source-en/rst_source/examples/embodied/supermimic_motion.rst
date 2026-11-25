G1 Motion Data Preparation
==========================

Prepare reference motion clips for the :doc:`G1 Motion Tracking Environment <g1_mimic>` and :doc:`SuperMimic <supermimic>`, including offline LAFAN1 conversion, the experimental BONES-SEED path, and custom ``clip_*.pt`` files.

**Prerequisites:** :doc:`Installation </rst_source/start/installation>` · the SuperMimic environment from :doc:`g1_mimic`.

Setup and Run
-------------

Activate the SuperMimic virtual environment from :doc:`g1_mimic` and run all commands from the RLinf repository root.

Offline LAFAN1 Conversion
-------------------------

If the training machine cannot reach Hugging Face, convert LAFAN1 on another machine that has network access, an RLinf checkout, and Python with ``torch``, ``numpy``, and ``omegaconf``. Because conversion checks against mjlab's G1 default joint pose, export that pose from the training machine first:

1. On the training machine, export the default joint pose to ``g1_q0.pt``:

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt

2. Copy ``g1_q0.pt`` to the conversion machine.
3. On the conversion machine, download and convert the 40 clips using that pose:

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \
          --q0-file /path/to/g1_q0.pt --accept-license

4. Copy the entire ``/data/lafan1_g1`` directory (including ``manifest.json``) to the training machine and point ``SUPERMIMIC_MOTION_PATH`` to it.

BONES-SEED (Experimental)
-------------------------

The public G1 retargeting lives in the Hugging Face dataset `bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_. Training uses ``g1.tar.gz`` and ``LICENSE.md`` from that dataset. Accept the license on the dataset page, read ``LICENSE.md``, then download and extract:

.. code-block:: bash

   huggingface-cli download bones-studio/seed g1.tar.gz LICENSE.md \
       --repo-type dataset --local-dir /data/bones-seed
   tar -xzf /data/bones-seed/g1.tar.gz -C /data/bones-seed

If the dataset asks you to sign in, run ``huggingface-cli login`` first. The CSVs are under ``g1/csv/`` after extraction.

Conversion resamples the 120 Hz CSVs to 50 Hz ``clip_*.pt`` files and keeps every clip:

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1 \
       --jobs 16
   export SUPERMIMIC_MOTION_PATH=/data/bones_g1

A successful run ends with:

.. code-block:: text

   [bones-seed] done: 142220 converted, 0 skipped, 0 failed

Loading keeps clips whose names match the terms named in Appendix E: 135,798 of 142,220. The paper's name filter keeps 129,667, and it does not publish the full term list.

A directory of ``clip_*.pt`` files opens every file on each launch. Point ``SUPERMIMIC_MOTION_PATH`` at a packed directory instead: ``joint_pos.npy``, ``joint_vel.npy``, the four root ``.npy`` fields, ``motion_ends.npy``, and ``joint_names.json``. ``root_ang_vel_w`` is stored in the world frame and converted to the body frame on load. Optional ``q0.npy`` is the 29-d default joint pose; without it the loader uses mjlab's G1 keyframe. Optional ``motion_names.json`` names the clips. That directory is not read as one file per clip.

The command below trains the GenCoord recipe for 20,000 updates.

.. code-block:: bash

   export RUN_DIR=/data/runs/bones_gencoord
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.logger.log_path=$RUN_DIR actor.cuda_graph=True \
       runner.max_epochs=20000 \
       env.train.motion_format=bones_seed env.train.num_clips=null \
       env.eval.motion_format=bones_seed

The log should report ``kept`` as 135798.

Evaluation plays each of the 135,798 kept clips from frame 0. The product of ``env.eval.total_num_envs`` and ``env.eval.rollout_epoch`` must be at least that count; 16,384 and 9 meet it, with the envs split across four GPUs. Set ``max_steps_per_rollout_epoch`` to the longest kept clip in control steps (frame count minus 1). A short limit stops before evaluation starts and prints ``the longest evaluation job runs <N> control steps``; rerun with that N.

.. code-block:: bash

   bash evaluations/run_eval.sh supermimic_lafan1_eval \
       runner.ckpt_path=$RUN_DIR/supermimic_lafan1_gencoord/checkpoints/global_step_20000/actor/model_state_dict/full_weights.pt \
       runner.eval_export.output_dir=$RUN_DIR/eval_20000 \
       env.eval.motion_format=bones_seed \
       env.eval.num_clips=null \
       env.eval.total_num_envs=16384 \
       env.eval.rollout_epoch=9 \
       env.eval.max_steps_per_rollout_epoch=<N> \
       '~cluster.component_placement' \
       '+cluster.component_placement={env:0-3,rollout:0-3}'

Run It
------

Save each custom clip in the ``clip_*.pt`` format below inside one directory (loaded in file-name order), point ``SUPERMIMIC_MOTION_PATH`` to that directory, and pass ``env.train.num_clips=null`` (or ``env.eval.num_clips=null`` for evaluation) to load every clip in the folder:

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/my_clips
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       env.train.num_clips=null \
       runner.logger.log_path=/data/runs/supermimic_custom

To evaluate more than 40 clips, adjust the evaluation capacity settings described in :doc:`../../evaluations/guides/supermimic`. If the directory includes a ``manifest.json``, the loader verifies every clip's SHA-256 and rejects unlisted files; without ``manifest.json``, it checks tensor shapes and values only.

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

Export ``g1_q0.pt`` once with ``python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt``, then save each custom clip dictionary with ``torch.save`` (for example ``clip_000.pt``). The training and evaluation loader validates every clip's keys, joint order, ``q0`` match, quaternion norms, and finite values on startup.
