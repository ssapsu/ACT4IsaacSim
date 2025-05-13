#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACTPolicy 학습 스크립트
- train_bc 함수에 tqdm 진행률 및 ETA 표시 추가
- 에폭별 학습/검증 손실을 기록하고, 최종에 학습곡선 플롯 저장
- Early Stopping 기능 추가
"""
import torch
import os
import pickle
import argparse
import time
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import load_data, set_seed
from policy import ACTPolicy

BOX_POSE = [None]  # to be changed from outside
DT = 0.02


def main(args):
    set_seed(args.seed)
    is_eval = args.eval
    ckpt_dir = args.ckpt_dir
    policy_class = args.policy_class
    onscreen_render = args.onscreen_render
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    patience = args.patience
    action_offset = args.action_offset

    camera_names = [c for c in ["color", "left_color", "arm_color"] if c in ["color", "left_color", "CCTV_color", "arm_color"]]

    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "backbone": "resnet18",
        "lr_backbone": 1e-5,
        "enc_layers": 8,
        "dec_layers": 8,
        "nheads": 16,
        "camera_names": camera_names,
    }
    config = {
        "ckpt_dir": ckpt_dir,
        "seed": args.seed,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "real_robot": False,
        "num_epochs": num_epochs,
        "patience": patience,
    }

    train_loader, val_loader, stats, _ = load_data(
        './dataset/isaac_sim_example', 50, camera_names,
        batch_size, batch_size, action_offset
    )

    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    if is_eval:
        print("Eval 모드는 아직 지원되지 않습니다.")
        return

    best, train_losses, val_losses = train_bc(train_loader, val_loader, config)
    epoch, loss, state = best
    torch.save(state, os.path.join(ckpt_dir, "policy_best.ckpt"))
    print(f"최적 에폭: {epoch}, 손실: {loss:.6f}")

    plot_losses(train_losses, val_losses, ckpt_dir)


def make_policy(pc, pcfg):
    if pc == "ACT":
        return ACTPolicy(pcfg)
    raise NotImplementedError


def make_optimizer(pc, policy):
    return policy.configure_optimizers()


def forward_pass(batch, policy):
    img, qpos, fut, mask = batch
    img, qpos, fut, mask = img.cuda(), qpos.cuda(), fut.cuda(), mask.cuda()
    return policy(qpos, img, fut, mask)


def train_bc(train_loader, val_loader, config):
    set_seed(config["seed"])
    policy = make_policy(config["policy_class"], config["policy_config"]).cuda()
    optimizer = make_optimizer(config["policy_class"], policy)

    best = (0, float("inf"), None)
    num_epochs = config.get("num_epochs", 1000)
    patience = config.get("patience", 10)
    no_improve = 0

    train_losses = []
    val_losses = []

    epoch_bar = tqdm(range(num_epochs), desc="Epochs", unit="ep")
    start_time = time.time()
    for epoch in epoch_bar:
        epoch_start = time.time()

        # Validation
        policy.eval()
        val_accum, val_count = 0.0, 0
        for batch in val_loader:
            out = forward_pass(batch, policy)
            val_accum += out["loss"].detach().cpu().mean().item()
            val_count += 1
        if val_count == 0:
            raise RuntimeError("Validation loader가 비어있습니다.")
        val_loss = val_accum / val_count
        val_losses.append(val_loss)

        # Early Stopping 체크
        if val_loss < best[1]:
            best = (epoch, val_loss, deepcopy(policy.state_dict()))
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"얼리 스탑핑: {patience} 에폭 동안 개선 없음. 종료.")
                break

        # Training
        policy.train()
        train_accum, train_batches = 0.0, 0
        for batch in train_loader:
            out = forward_pass(batch, policy)
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()
            train_accum += out["loss"].item()
            train_batches += 1
        train_loss = train_accum / train_batches if train_batches > 0 else float("nan")
        train_losses.append(train_loss)

        # 에폭별 시간 및 ETA 계산
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        remaining = (num_epochs - (epoch + 1)) * (elapsed / (epoch + 1))

        # tqdm 정보 업데이트
        epoch_bar.set_postfix({
            "val_loss": f"{val_loss:.4f}",
            "train_loss": f"{train_loss:.4f}",
            "epoch_time(s)": f"{epoch_time:.2f}",
            "ETA(s)": f"{remaining:.1f}"
        })

    return best, train_losses, val_losses


def plot_losses(train_losses, val_losses, ckpt_dir):
    """학습/검증 손실 곡선을 그려서 저장합니다."""
    # Early stopping으로 인해 길이가 다를 수 있으므로 최소 길이 사용
    n = min(len(train_losses), len(val_losses))
    if n == 0:
        print("[Warning] 기록된 에폭이 없습니다. 플롯 생략.")
        return
    epochs = range(1, n + 1)
    plt.figure()
    plt.plot(epochs, train_losses[:n], label="Train Loss")
    plt.plot(epochs, val_losses[:n], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(ckpt_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Info] 손실 곡선을 저장했습니다: {save_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true")
    p.add_argument("--onscreen_render", action="store_true")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--policy_class", required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num_epochs", type=int, required=True)
    p.add_argument("--patience", type=int, default=100, help="Early stopping patience")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--chunk_size", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--action_offset", type=int, default=30)
    args = p.parse_args()
    main(args)
