#!/usr/bin/env bash
# Fine-tune the GraspGen Robotiq 2F-140 discriminator on our ISCAR dataset.
# Reuses GraspGen's training infra (scripts/train_graspgen.py); only paths,
# checkpoint, and a couple of fine-tune-flavoured hyperparams differ.
set -euo pipefail

# Use bp_runtime/ros_venv — has torch 2.8 cu129 (Blackwell-ready) + hydra +
# grasp_gen editable. The eomt conda env doesn't have hydra/h5py/meshcat etc.
ROS_VENV=/home/kaelin/bp_runtime/ros_venv
export PATH="${ROS_VENV}/bin:${PATH}"

GRASPGEN_DIR=/home/kaelin/bp_runtime/ml_deps/GraspGen
DATA_DIR=/home/kaelin/bp_runtime/ml_deps/eomt/grasp_finetune_data
RESULTS_DIR=/home/kaelin/bp_runtime/ml_deps/eomt/grasp_finetune_results

GRIPPER=robotiq_2f_140
PRETRAINED=${GRASPGEN_DIR}/models/checkpoints/graspgen_${GRIPPER}_dis.pth
LOG_DIR=${RESULTS_DIR}/logs/${GRIPPER}_dis_finetune
CACHE_DIR=${RESULTS_DIR}/cache

mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

# Read the pretrained checkpoint's epoch counter: train_graspgen.py:454 loops
# `range(init_epoch, num_epochs)`, so num_epochs MUST be larger than the
# pretrained ckpt's epoch or the loop runs zero iterations (silent no-op).
START_EPOCH=$(python -c "import torch, sys; \
    ck = torch.load('${PRETRAINED}', map_location='cpu', weights_only=False); \
    print(ck.get('epoch', 0))")
TARGET_EPOCH=${TARGET_EPOCH_OVERRIDE:-$((START_EPOCH + 300))}
echo "[finetune_dis] pretrained ckpt epoch=${START_EPOCH}, fine-tuning to epoch=${TARGET_EPOCH}"

# Resume from local last.pth if any (mid-run continuation); otherwise from pretrained.
CHECKPOINT=${LOG_DIR}/last.pth
[ -f "${CHECKPOINT}" ] || CHECKPOINT=${PRETRAINED}
echo "[finetune_dis] using checkpoint ${CHECKPOINT}"

# Pin to one GPU; osmesa needed for the pyrender partial-PC augmentation path.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYOPENGL_PLATFORM=osmesa

cd "${GRASPGEN_DIR}/scripts" && python train_graspgen.py \
    data.num_points=2048 \
    data.load_contact=False \
    data.dataset_cls="ObjectPickDataset" \
    data.rotation_augmentation=True \
    data.root_dir="${DATA_DIR}/splits/${GRIPPER}" \
    data.object_root_dir="${DATA_DIR}/object_dataset" \
    data.grasp_root_dir="${DATA_DIR}/grasp_data/${GRIPPER}" \
    data.dataset_name=iscar \
    data.dataset_version=v2 \
    data.prob_point_cloud=-1 \
    data.redundancy=7 \
    data.gripper_name=${GRIPPER} \
    data.cache_dir="${CACHE_DIR}" \
    data.num_grasps_per_object=300 \
    data.preload_dataset=False \
    data.load_discriminator_dataset=True \
    data.discriminator_ratio="[0.45,0.45,0.00,0.10,0.00,0.00,0.00]" \
    train.log_dir="${LOG_DIR}" \
    train.batch_size=16 \
    train.num_gpus=1 \
    train.num_epochs=${TARGET_EPOCH} \
    train.num_workers=4 \
    train.print_freq=10 \
    train.plot_freq=50 \
    train.save_freq=10 \
    train.eval_freq=5 \
    train.checkpoint="${CHECKPOINT}" \
    train.model_name='discriminator' \
    optimizer.type="ADAMW" \
    optimizer.grad_clip=-1 \
    optimizer.lr=1e-5 \
    discriminator.gripper_name=${GRIPPER} \
    discriminator.topk_ratio=0.75 \
    discriminator.obs_backbone=pointnet \
    discriminator.grasp_repr=r3_so3 \
    discriminator.pose_repr=mlp \
    discriminator.kappa=2.02217 \
    2>&1 | tee "${LOG_DIR}/console_log.txt"
