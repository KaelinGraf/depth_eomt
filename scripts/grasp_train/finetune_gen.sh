#!/usr/bin/env bash
# Fine-tune the GraspGen Robotiq 2F-140 diffusion generator on our ISCAR dataset.
# Mirrors finetune_dis.sh — only model_name and the diffusion-specific block change.
set -euo pipefail

# Use bp_runtime/ros_venv — has torch 2.8 cu129 (Blackwell-ready) + hydra +
# grasp_gen editable. The eomt conda env doesn't have hydra/h5py/meshcat etc.
ROS_VENV=/home/kaelin/bp_runtime/ros_venv
export PATH="${ROS_VENV}/bin:${PATH}"

GRASPGEN_DIR=/home/kaelin/bp_runtime/ml_deps/GraspGen
DATA_DIR=/home/kaelin/bp_runtime/ml_deps/eomt/grasp_finetune_data
RESULTS_DIR=/home/kaelin/bp_runtime/ml_deps/eomt/grasp_finetune_results

GRIPPER=robotiq_2f_140
PRETRAINED=${GRASPGEN_DIR}/models/checkpoints/graspgen_${GRIPPER}_gen.pth
LOG_DIR=${RESULTS_DIR}/logs/${GRIPPER}_gen_finetune
CACHE_DIR=${RESULTS_DIR}/cache

mkdir -p "${LOG_DIR}" "${CACHE_DIR}"

START_EPOCH=$(python -c "import torch; \
    ck = torch.load('${PRETRAINED}', map_location='cpu', weights_only=False); \
    print(ck.get('epoch', 0))")
TARGET_EPOCH=${TARGET_EPOCH_OVERRIDE:-$((START_EPOCH + 500))}
echo "[finetune_gen] pretrained ckpt epoch=${START_EPOCH}, fine-tuning to epoch=${TARGET_EPOCH}"

CHECKPOINT=${LOG_DIR}/last.pth
[ -f "${CHECKPOINT}" ] || CHECKPOINT=${PRETRAINED}
echo "[finetune_gen] using checkpoint ${CHECKPOINT}"

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
    data.num_grasps_per_object=500 \
    data.load_discriminator_dataset=False \
    data.visualize_batch=False \
    train.log_dir="${LOG_DIR}" \
    train.batch_size=16 \
    train.num_gpus=1 \
    train.num_epochs=${TARGET_EPOCH} \
    train.num_workers=8 \
    train.print_freq=10 \
    train.plot_freq=50 \
    train.save_freq=10 \
    train.eval_freq=5 \
    train.checkpoint="${CHECKPOINT}" \
    train.model_name='diffusion' \
    optimizer.type="ADAMW" \
    optimizer.grad_clip=-1 \
    optimizer.lr=1e-5 \
    diffusion.gripper_name=${GRIPPER} \
    diffusion.num_diffusion_iters=10 \
    diffusion.num_diffusion_iters_eval=10 \
    diffusion.obs_backbone=pointnet \
    diffusion.grasp_repr=r3_so3 \
    diffusion.attention='cat_attn' \
    diffusion.compositional_schedular=True \
    diffusion.loss_pointmatching=False \
    diffusion.loss_l1_pos=True \
    diffusion.loss_l1_rot=True \
    diffusion.ptv3.grid_size=0.01 \
    diffusion.pose_repr='mlp' \
    diffusion.kappa=2.02217 \
    2>&1 | tee "${LOG_DIR}/console_log.txt"
