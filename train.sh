# python3 train.py --model_type compression --regime med --n_steps 1e5
export CUDA_VISIBLE_DEVICES=1
python3 train.py --model_type compression_gan --regime med --n_steps 1e5 --warmstart -ckpt /eva_data1/thchou1006/compression/high-fidelity-generative-compression/experiments/coco_compression_2023_12_23_16_57/ckeckpoints/coco_compression_2023_12_23_16_57_epoch6_idx100001_2023_12_24_05:36.pt