import torch
import os
import pickle
import argparse
from copy import deepcopy

from utils import load_data, set_seed
from policy import ACTPolicy

BOX_POSE = [None]  # to be changed from outside
DT = 0.02


def main(args):
    # 설정
    set_seed(args.seed)
    is_eval = args.eval
    ckpt_dir = args.ckpt_dir
    policy_class = args.policy_class
    onscreen_render = args.onscreen_render
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    action_offset = args.action_offset

    # 태스크 로드
    is_sim = True
    if is_sim:
        from constants import SIM_TASK_CONFIGS as CONFIGS
    # RGB 카메라만 사용
    camera_names = [c for c in ["color", "left_color"] if c in ["color", "left_color"]]

    # 정책 구성
    backbone_lr = 1e-5
    policy_config = {
        "lr": args.lr,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "lr_backbone": backbone_lr,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 7,
        "nheads": 8,
        "camera_names": camera_names,
    }
    config = {
        "ckpt_dir": ckpt_dir,
        "seed": args.seed,
        "policy_class": policy_class,
        "policy_config": policy_config,
        "real_robot": not is_sim,
    }

    # 데이터 로드
    train_loader, val_loader, stats, _ = load_data(
        './dataset/isaac_sim_example', 3, camera_names, batch_size, batch_size, action_offset
    )

    # 정규화 통계 저장
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "wb") as f:
        pickle.dump(stats, f)

    # 평가 모드
    if is_eval:
        # eval_bc 함수 필요시 구현
        print("Eval 모드는 아직 지원되지 않습니다.")
        return

    # 학습
    best = train_bc(train_loader, val_loader, config)
    epoch, loss, state = best
    torch.save(state, os.path.join(ckpt_dir, "policy_best.ckpt"))
    print(f"최적 에폭: {epoch}, 손실: {loss:.6f}")


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

    for epoch in range(config.get("num_epochs", 1000)):
        # Validation (running mean)
        policy.eval()
        accum = {}
        count = 0
        for batch in val_loader:
            out = forward_pass(batch, policy)
            # batch별 텐서 평균 후 CPU로 변환
            for k, v in out.items():
                val = v.detach().cpu().mean().item()
                accum[k] = accum.get(k, 0.0) + val
            count += 1
        if count == 0:
            raise RuntimeError("Validation loader가 비어있습니다.")
        vs = {k: accum[k] / count for k in accum}
        if vs.get("loss", float("inf")) < best[1]:
            best = (epoch, vs["loss"], deepcopy(policy.state_dict()))

        # Training
        policy.train()
        for batch in train_loader:
            out = forward_pass(batch, policy)
            out["loss"].backward()
            optimizer.step()
            optimizer.zero_grad()

    return best


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval", action="store_true")
    p.add_argument("--onscreen_render", action="store_true")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--policy_class", required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num_epochs", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--chunk_size", type=int, default=1)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--action_offset", type=int, default=20)
    args = p.parse_args()
    main(args)
