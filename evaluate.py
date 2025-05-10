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
    reader = EpisodeReader(
        os.path.join("./dataset/isaac_sim_example", "episode_0.hdf5")
    )
    T = len(reader)
    num_q = args.chunk_size
    state_dim = stats["qpos_mean"].shape[0]  # action_dim 가져오기

    # chunk_size > 1 이면 temporal aggregation 으로 간주
    if num_q > 1:
        # (T, T+num_q, state_dim) 버퍼
        buffer = torch.zeros(T, T + num_q, state_dim, device="cuda")

    # 5) inference 루프
    with torch.inference_mode():
        for t, (imgs, qpos) in enumerate(reader):
            # normalize & tensor 변환
            q = torch.from_numpy(pre(qpos)).float().cuda().unsqueeze(0)
            cams = []
            for cam in args.camera_names:
                im = rearrange(imgs[cam], "H W C -> C H W")
                cams.append(im)
            bimg = torch.from_numpy(np.stack(cams) / 255.0).float().cuda().unsqueeze(0)

            # 모델 추론 (ACTPolicy: (1, num_q, state_dim))
            raw_out = model(q, bimg)  # tensor

            if num_q > 1:
                # 버퍼에 저장
                buffer[t, t : t + num_q] = raw_out.squeeze(0)
                # temporal_aggregate 호출
                agg = temporal_aggregate(buffer, t, num_q)
                action = agg.squeeze(0).cpu().numpy()
                print(f"t={t}, aggregated action:", action)
            else:
                # 그냥 첫 쿼리 사용
                action = raw_out[0, 0].cpu().numpy()
                print(f"t={t}, raw action:", action)


def temporal_aggregate(
    all_time_actions: torch.Tensor, t: int, num_queries: int, k: float = 0.01
) -> torch.Tensor:
    """
    시간축 앙상블 (exponential weighted average)
    all_time_actions: (T, T+num_queries, state_dim)
    t: 현재 타임스텝
    num_queries: 쿼리 수
    k: 감쇠 계수
    """
    acts = all_time_actions[:, t]  # (T, state_dim)
    mask = torch.all(acts != 0, dim=1)  # valid rows
    valid = acts[mask]  # (N_valid, state_dim)
    if valid.numel() == 0:
        # 아직 유효 액션이 없으면 0 벡터 반환
        return torch.zeros(1, all_time_actions.size(-1), device=all_time_actions.device)
    n = valid.size(0)
    weights = torch.exp(-k * torch.arange(n, device=valid.device, dtype=torch.float32))
    weights = weights / weights.sum()  # (N_valid,)
    return (valid * weights.unsqueeze(1)).sum(dim=0, keepdim=True)  # (1, state_dim)


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
