G1 动作数据准备
===============

为 :doc:`G1 动作跟踪环境 <g1_mimic>` 和 :doc:`SuperMimic <supermimic>` 准备参考动作片段，包括 LAFAN1 离线转换、实验性 BONES-SEED 以及自定义 ``clip_*.pt`` 文件格式。

**前置条件：** :doc:`安装 </rst_source/start/installation>` · :doc:`G1 动作跟踪环境 <g1_mimic>` 中的 SuperMimic 环境。

安装与运行
----------

先激活 :doc:`g1_mimic` 中创建的 SuperMimic 虚拟环境，以下所有命令均在 RLinf 仓库根目录执行。

离线转换 LAFAN1
---------------

训练机器无法访问 Hugging Face 时，可在另一台联网机器上完成下载和转换。转换机器需要 RLinf 代码以及装有 ``torch``、``numpy`` 和 ``omegaconf`` 的 Python 环境。由于转换会校验 mjlab G1 的默认关节姿态，需要先在训练机器上导出默认关节姿态：

1. 在训练机器上导出默认关节姿态到 ``g1_q0.pt``：

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt

2. 将 ``g1_q0.pt`` 复制到转换机器。
3. 在转换机器上使用这个姿态文件，下载并转换 40 条片段：

   .. code-block:: bash

      python toolkits/supermimic/prepare_lafan1.py --out /data/lafan1_g1 \
          --q0-file /path/to/g1_q0.pt --accept-license

4. 将整个 ``/data/lafan1_g1`` 目录（含 ``manifest.json``）复制到训练机器，并将 ``SUPERMIMIC_MOTION_PATH`` 指向这个目录。

BONES-SEED（实验性）
--------------------

公开的 G1 重定向数据在 Hugging Face 数据集 `bones-studio/seed <https://huggingface.co/datasets/bones-studio/seed>`_。训练用其中的 ``g1.tar.gz`` 和 ``LICENSE.md``。在网页上接受许可并阅读 ``LICENSE.md`` 后下载、解压：

.. code-block:: bash

   huggingface-cli download bones-studio/seed g1.tar.gz LICENSE.md \
       --repo-type dataset --local-dir /data/bones-seed
   tar -xzf /data/bones-seed/g1.tar.gz -C /data/bones-seed

数据集要求登录时，先执行 ``huggingface-cli login``。CSV 位于解压后的 ``g1/csv/``。

转换脚本把 120 Hz 的 CSV 重采样为 50 Hz 的 ``clip_*.pt``，这一步保留全部片段：

.. code-block:: bash

   python toolkits/supermimic/convert_bones_seed.py \
       --input /data/bones-seed/g1/csv \
       --output /data/bones_g1 \
       --jobs 16
   export SUPERMIMIC_MOTION_PATH=/data/bones_g1

成功时最后一行是：

.. code-block:: text

   [bones-seed] done: 142220 converted, 0 skipped, 0 failed

加载时按附录 E 点名的词过滤，142,220 条留下 135,798 条。论文的名字过滤留下 129,667 条，完整词表没有公布。

``clip_*.pt`` 目录每次启动都会逐个打开文件再过滤。把筛过的片段打成官方那种目录后，把 ``SUPERMIMIC_MOTION_PATH`` 指到该目录即可：里面是 ``joint_pos.npy``、``joint_vel.npy``、四个根状态 ``.npy``、``motion_ends.npy`` 和 ``joint_names.json``。``root_ang_vel_w`` 按世界系存放，加载时转回身体系。可选的 ``q0.npy`` 是 29 维默认关节角，没有则用 mjlab 的 G1 关键帧。可选的 ``motion_names.json`` 是片段名。这种目录不再逐个读 ``clip_*.pt``。

下面在 GenCoord 配置上训练 20,000 次更新。

.. code-block:: bash

   export RUN_DIR=/data/runs/bones_gencoord
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       runner.logger.log_path=$RUN_DIR actor.cuda_graph=True \
       runner.max_epochs=20000 \
       env.train.motion_format=bones_seed env.train.num_clips=null \
       env.eval.motion_format=bones_seed

日志里的 ``kept`` 应为 135798。

评测从第 0 帧播放这 135,798 条片段。``env.eval.total_num_envs`` 与 ``env.eval.rollout_epoch`` 的乘积要不少于这个数量，下面的 16,384 和 9 满足该条件，由四张 GPU 分担环境。``max_steps_per_rollout_epoch`` 取保留片段里最长的控制步数（帧数减 1）。小于最长片段时，评测会在开始前退出，报错里的 ``the longest evaluation job runs <N> control steps`` 给出这个步数，改成 N 再启动。

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

这次训练的结果
~~~~~~~~~~~~~~

名字过滤留下 135,798 条片段。用 4 卡、bf16、GenCoord 对这批片段训了 80,000 次更新，中间存了 20,000 和 80,000 两个权重，两个都做了全库评测，结果如下：

.. list-table::
   :header-rows: 1
   :widths: 14 22 18 24 22

   * - 更新次数
     - ``normalized_return``
     - 摔倒
     - ``body_pos_err``\ （m）
     - ``torso_ori_err``
   * - 80000
     - 0.937
     - 421（0.310%）
     - 0.021
     - 0.079
   * - 20000
     - 0.927
     - 830（0.611%）
     - 0.023
     - —

80,000 的存档上，135,798 条里有 0.997 播完，平均一条 7.34 秒。

训练时记下的数如下。单步跟踪奖励最高是 0.1。

.. list-table::
   :header-rows: 1
   :widths: 16 18 20 22 24

   * - 更新次数
     - 单步奖励
     - 摔倒比例
     - actor 回报
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

运行
----

按下方格式写出每条自定义 ``clip_*.pt`` 并放入同一目录（加载器按文件名排序读取），将 ``SUPERMIMIC_MOTION_PATH`` 指向该目录，训练时加 ``env.train.num_clips=null``\ （评测加 ``env.eval.num_clips=null``）即可加载目录下全部片段：

.. code-block:: bash

   export SUPERMIMIC_MOTION_PATH=/data/my_clips
   python examples/embodiment/train_embodied_agent.py \
       --config-name supermimic_lafan1_gencoord \
       env.train.num_clips=null \
       runner.logger.log_path=/data/runs/supermimic_custom

评测超过 40 条片段时的容量配置见 :doc:`../../evaluations/guides/supermimic`。如果目录包含 ``manifest.json``，加载器会校验每条片段的 SHA-256 并拒绝未登记文件；不含 ``manifest.json`` 时仅检查张量形状与数值合法性。

片段文件格式
------------

每个 ``clip_*.pt`` 都是用 ``torch.save`` 保存的 Python 字典：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 键
     - 内容
   * - ``fps``
     - ``50`` 或 ``30``。30 Hz 片段会自动重采样到 50 Hz 控制网格并用中心差分重算速度；50 Hz 片段按原样读取，**速度列必须填写准确**。不支持其他帧率。
   * - ``joint_names``
     - 29 个关节名列表，顺序必须与 mjlab Unitree G1 一致（左腿、右腿、腰部、左臂、右臂）。
   * - ``q0``
     - 29 维默认关节姿态（rad），必须与 mjlab 站立默认姿态一致。
   * - ``state``
     - 形状为 ``[T, 71]`` 的浮点张量，列布局见下表；不能包含 NaN/Inf，根四元数必须已归一化。

``state``\ （``[T, 71]``）列布局如下：

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - 列区间
     - 内容
     - 单位
   * - ``0:3``
     - 根节点（骨盆）位置，世界坐标系
     - m
   * - ``3:7``
     - 根节点四元数（``wxyz``）
     - —
   * - ``7:36``
     - 关节角度，顺序同 ``joint_names``
     - rad
   * - ``36:39``
     - 根节点线速度，世界坐标系
     - m/s
   * - ``39:42``
     - 根节点角速度，根节点坐标系
     - rad/s
   * - ``42:71``
     - 关节角速度
     - rad/s

先用 ``python toolkits/supermimic/prepare_lafan1.py --export-q0 g1_q0.pt`` 导出默认关节姿态 ``g1_q0.pt``，再用 ``torch.save`` 将每条自定义片段字典保存为 ``clip_000.pt`` 等文件。训练与评测启动时，加载器会自动校验字段、关节顺序、``q0`` 一致性、四元数模长和数值有限性。
