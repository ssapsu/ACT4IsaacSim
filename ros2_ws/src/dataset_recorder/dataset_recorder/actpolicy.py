#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACTPolicy 기반 ROS2 실시간 추론 노드
- argparse로 설정값(ckpt_dir, camera_names 등) 입력
- /camera, /odom, /joint_states 토픽에서 데이터 수집
- 설정된 주기(sample_rate)마다 모델 추론
- 터미널에 결과 출력
"""
import os
import argparse
import pickle
from collections import deque

import cv2
import torch
import numpy as np
from einops import rearrange

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

from dataset_recorder.policy import ACTPolicy
from dataset_recorder.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser("Custom ACT evaluator")
    # 필수 인자
    parser.add_argument(
        "--ckpt_dir", required=True, default="/home/hyeonsu/Documents/ACT4IsaacSim/ckpt"
    )
    parser.add_argument("--ckpt_name", default="policy_best.ckpt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_names", nargs="+", default=["color", "left_color"])
    # ACT 하이퍼파라
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--policy_class", type=str, default="ACT")
    parser.add_argument("--num_epochs", type=int, default=1)
    # ROS2 노드 추가 파라미터
    parser.add_argument("--sample_rate", type=float, default=10.0)
    parser.add_argument("--stats_file", default="dataset_stats.pkl")
    return parser.parse_args()


def temporal_aggregate(
    all_time_actions: torch.Tensor, t: int, num_queries: int, k: float = 0.01
) -> torch.Tensor:
    acts = all_time_actions[:, t]
    mask = torch.all(acts != 0, dim=1)
    valid = acts[mask]
    if valid.numel() == 0:
        return torch.zeros(1, all_time_actions.size(-1), device=all_time_actions.device)
    n = valid.size(0)
    weights = torch.exp(-k * torch.arange(n, device=valid.device, dtype=torch.float32))
    weights = weights / weights.sum()
    return (valid * weights.unsqueeze(1)).sum(dim=0, keepdim=True)


class AIInferenceNode(Node):
    def __init__(self, args):
        super().__init__("ai_inference_node")
        self.args = args
        # 1) 시드 고정
        set_seed(args.seed)
        # 2) 모델 로드
        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        policy_cfg = {
            "lr": 1e-5,
            "num_queries": args.chunk_size,
            "kl_weight": 10.0,
            "hidden_dim": 512,
            "dim_feedforward": 3200,
            "backbone": "resnet18",
            "lr_backbone": 1e-5,
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": args.camera_names,
            "policy_class": "ACT",
            "num_epochs": 1,
        }
        self.model = ACTPolicy(policy_cfg)
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.cuda().eval()
        # 3) stats 로드
        stats_path = os.path.join(args.ckpt_dir, args.stats_file)
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        self.qpos_mean = torch.tensor(stats["qpos_mean"], dtype=torch.float32).cuda()
        self.qpos_std = torch.tensor(stats["qpos_std"], dtype=torch.float32).cuda()
        # 4) 브리지와 버퍼 초기화
        self.bridge = CvBridge()
        self.img_buf = {cam: None for cam in args.camera_names}
        self.img_stamp = None
        self.odom_buf = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.joint_initialized = False
        if args.chunk_size > 1:
            T = 10000
            dim = self.qpos_mean.numel()
            self.action_buffer = torch.zeros(
                (T, T + args.chunk_size, dim), device="cuda"
            )
            self.t_idx = 0
        # 5) 토픽 구독
        qos = 10
        for cam in args.camera_names:
            self.create_subscription(
                Image, f"/camera/{cam}", lambda msg, c=cam: self.img_cb(c, msg), qos
            )
        self.create_subscription(Odometry, "/odom", self.odom_cb, qos)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, qos)
        # 6) 추론 타이머
        self.create_timer(1.0 / args.sample_rate, self.timer_cb)
        self.get_logger().info(
            f"Inference started: {args.camera_names} @ {args.sample_rate}Hz"
        )

    def img_cb(self, cam, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.img_buf[cam] = (ts, rgb)
        self.img_stamp = ts

    def odom_cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        self.odom_buf.append((ts, (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z)))

    def joint_cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joint_buf.append((ts, list(msg.name), list(msg.position)))

    def _pop(self, buf, ts, tol=0.05):
        if not buf:
            return None
        diffs = [abs(t - ts) for t, *_ in buf]
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tol:
            item = buf[idx]
            del buf[idx]
            return item
        return None

    def timer_cb(self):
        if any(v is None for v in self.img_buf.values()):
            return
        ts = self.img_stamp
        cams = [
            rearrange(self.img_buf[c][1], "H W C -> C H W")
            for c in self.args.camera_names
        ]
        img = torch.from_numpy(np.stack(cams) / 255.0).float().cuda().unsqueeze(0)
        od = self._pop(self.odom_buf, ts)
        j = self._pop(self.joint_buf, ts)
        if j and not self.joint_initialized:
            _, names, poses = j
            idxs = [names.index(f"joint_{i}") for i in range(1, 7)] + [
                names.index("Slider_1")
            ]
            self.joint_idxs = idxs
            self.joint_initialized = True
        pos_sel = (
            np.array(j[2])[self.joint_idxs]
            if j and self.joint_initialized
            else np.zeros(len(self.joint_idxs) if hasattr(self, "joint_idxs") else 7)
        )
        lin_sp = np.linalg.norm(np.array(od[1]) if od else np.zeros(3))
        ang_sp = np.linalg.norm(np.array(od[2]) if od else np.zeros(3))
        qvec = np.concatenate([pos_sel, [lin_sp, ang_sp]]).astype(np.float32)
        q = (torch.from_numpy(qvec).cuda().float() - self.qpos_mean) / self.qpos_std
        q = q.unsqueeze(0)
        with torch.inference_mode():
            out = self.model(q, img)
            if self.args.chunk_size > 1:
                t = self.t_idx
                self.action_buffer[t, t : t + self.args.chunk_size] = out.squeeze(0)
                action = (
                    temporal_aggregate(self.action_buffer, t, self.args.chunk_size)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
                self.t_idx += 1
            else:
                action = out[0, 0].cpu().numpy()
        print(f'[Inference] t={getattr(self,"t_idx",0)}, action={action}')


def main():
    rclpy.init()
    args = parse_args()
    node = AIInferenceNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
