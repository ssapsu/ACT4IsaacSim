#!/usr/bin/env python3
import os
import pickle
import argparse

import torch
import numpy as np
import h5py
from einops import rearrange

from policy import ACTPolicy
from utils import set_seed


class EpisodeReader:
    def __init__(self, hdf5_path):
        self.f = h5py.File(hdf5_path, "r")
        grp = self.f.get("root", self.f)
        self.imgs = grp["observations/images"]
        self.qpos = grp["observations/qpos"]
        self.length = self.qpos.shape[0]
        self.idx = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.length:
            self.f.close()
            raise StopIteration
        t = self.idx
        images = {k: np.array(self.imgs[k][t]) for k in self.imgs.keys()}
        qpos = np.array(self.qpos[t], dtype=np.float32)
        self.idx += 1
        return images, qpos


def make_policy(policy_cfg):
    return ACTPolicy(policy_cfg)


def eval_episode(args):
    # 1) 시드 고정
    set_seed(args.seed)
    # 2) 체크포인트 로드
    ckpt = os.path.join(args.ckpt_dir, args.ckpt_name)
    model = make_policy(args.policy_cfg)
    model.load_state_dict(torch.load(ckpt))
    model.cuda().eval()

    # 3) stats 로드
    with open(os.path.join(args.ckpt_dir, "dataset_stats.pkl"), "rb") as f:
        stats = pickle.load(f)
    pre = lambda x: (x - stats["qpos_mean"]) / stats["qpos_std"]

    # 4) episode reader
    reader = EpisodeReader(os.path.join("./dataset/isaac_sim_example", "episode_0.hdf5"))

    # 5) inference 루프
    with torch.inference_mode():
        for imgs, qpos in reader:
            # normalize & tensor 변환
            q = torch.from_numpy(pre(qpos)).float().cuda().unsqueeze(0)
            cams = []
            for cam in args.camera_names:
                im = rearrange(imgs[cam], "H W C -> C H W")
                cams.append(im)
            bimg = torch.from_numpy(np.stack(cams) / 255.0).float().cuda().unsqueeze(0)

            raw = model(q, bimg).squeeze(0).cpu().numpy()
            print("action:", raw)


def main():
    parser = argparse.ArgumentParser("Custom ACT evaluator")
    # 필수 인자
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--ckpt_name", default="policy_best.ckpt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_names", default=["color", "left_color"])
    # ACT 하이퍼파라
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--policy_class", type=str, default="ACT")
    parser.add_argument("--num_epochs", type=int, default=1)

    args = parser.parse_args()

    # ACTPolicy에 넘길 설정 딕셔너리
    args.policy_cfg = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "backbone": "resnet18",
        "lr_backbone": 1e-5,
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": args.camera_names,
        "policy_class": "ACT",
        "num_epochs": 1,
    }

    eval_episode(args)


if __name__ == "__main__":
    main()
