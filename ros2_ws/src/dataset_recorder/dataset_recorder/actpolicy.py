#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACTPolicy 기반 ROS2 실시간 추론 노드
- argparse로 설정값(ckpt_dir, camera_names 등) 입력
- /camera, /odom, /joint_states 토픽에서 데이터 수집
- 설정된 주기(sample_rate)마다 모델 추론
- 타임 앙상블(과거 chunk_size 히스토리 가중 평균)으로 단일 액션 발행
- joint_command, cmd_vel 토픽에 발행
"""
import os
import sys
import time
import threading
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
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# 학습 스크립트 경로
sys.path.append("/home/hyeonsu/Documents/ACT4IsaacSim")
from policy import ACTPolicy
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser("Custom ACT evaluator")
    parser.add_argument("--ckpt_dir", default="/home/hyeonsu/Documents/ACT4IsaacSim/ckpt")
    parser.add_argument("--ckpt_name", default="policy_best.ckpt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_names", nargs="+", default=["color", "left_color", "depth"])
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--sample_rate", type=float, default=10.0)
    parser.add_argument("--stats_file", default="dataset_stats.pkl")
    return parser.parse_args()


def temporal_aggregate(all_time_actions: torch.Tensor, t: int, num_queries: int, k: float = 0.01) -> torch.Tensor:
    # 과거 예측들에서 column t를 모아 지수 가중 평균
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
        set_seed(args.seed)

        # 모델 로드
        ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
        policy_cfg = {
            "lr": 1e-5,
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
        }
        self.model = ACTPolicy(policy_cfg)
        self.model.load_state_dict(torch.load(ckpt_path))
        self.model.cuda().eval()

        # 정규화 통계 로드
        stats_path = os.path.join(args.ckpt_dir, args.stats_file)
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        if "joint_mean" in stats:
            jm = torch.tensor(stats["joint_mean"], dtype=torch.float32).cuda()
            js = torch.tensor(stats["joint_std"], dtype=torch.float32).cuda()
            vm = torch.tensor(stats["vel_mean"], dtype=torch.float32).cuda()
            vs = torch.tensor(stats["vel_std"], dtype=torch.float32).cuda()
            self.qpos_mean = torch.cat([jm, vm])
            self.qpos_std  = torch.cat([js, vs])
        else:
            self.qpos_mean = torch.tensor(stats["qpos_mean"], dtype=torch.float32).cuda()
            self.qpos_std  = torch.tensor(stats["qpos_std"], dtype=torch.float32).cuda()

        # 퍼블리셔 생성
        self.joint_pub = self.create_publisher(JointState, 'joint_command', 10)
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # CV Bridge 및 입력 버퍼
        self.bridge = CvBridge()
        self.img_buf = {c: None for c in args.camera_names}
        self.img_stamp = None
        self.odom_buf = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.joint_initialized = False

        # 시간 앙상블 버퍼 초기화
        T = 10000
        dim = self.qpos_mean.numel()
        # 버퍼 크기: (T 시점 × (T+chunk_size) 열 × dim)
        self.action_buffer = torch.zeros((T, T + args.chunk_size, dim), device="cuda")
        self.t_idx = 0

        # 토픽 구독
        qos = 10
        for cam in args.camera_names:
            self.create_subscription(
                Image,
                f"/camera/{cam}",
                lambda msg, c=cam: self.img_cb(c, msg),
                qos
            )
        self.create_subscription(Odometry, "/odom", self.odom_cb, qos)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, qos)

        # 추론 타이머
        interval = 1.0 / args.sample_rate
        self.create_timer(interval, self.timer_cb)
        self.get_logger().info(f"Inference node started @ {args.sample_rate}Hz")

    def img_cb(self, cam, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if cam == 'depth':
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')  # float32
            dmin, dmax = np.nanmin(depth), np.nanmax(depth)
            norm = (depth - dmin) / (dmax - dmin + 1e-6)
            # 3채널 uint8
            rgb = (norm * 255).astype(np.uint8)
            rgb = np.stack([rgb]*3, axis=2)
        else:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.img_buf[cam] = (ts, rgb)
        self.img_stamp = ts

    def odom_cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        self.odom_buf.append((ts, (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z)))

    def joint_cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joint_buf.append((ts, msg.name, msg.position))

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
        # 모든 카메라 이미지 준비 확인
        if any(v is None for v in self.img_buf.values()):
            return
        ts = self.img_stamp
        # 이미지 텐서 생성 (batch=1)
        cams = [rearrange(self.img_buf[c][1], "H W C -> C H W") for c in self.args.camera_names]
        img = torch.from_numpy(np.stack(cams) / 255.0).float().cuda().unsqueeze(0)

        # 동기화된 odom/joint 가져오기
        od = self._pop(self.odom_buf, ts)
        j  = self._pop(self.joint_buf, ts)
        if j and not self.joint_initialized:
            _, names, poses = j
            idxs = [names.index(f"joint_{i}") for i in range(1,7)] + [names.index("Slider_1")]
            self.joint_idxs = idxs
            self.joint_names = [names[i] for i in idxs]
            self.joint_initialized = True

        # qpos 벡터 생성 및 정규화
        if j and self.joint_initialized:
            _, _, poses = j
            pos_sel = np.array(poses)[self.joint_idxs]
        else:
            pos_sel = np.zeros(len(self.joint_names))
        lin_sp = np.linalg.norm(np.array(od[1]) if od else np.zeros(3))
        ang_sp = np.linalg.norm(np.array(od[2]) if od else np.zeros(3))
        qvec = np.concatenate([pos_sel, [lin_sp, ang_sp]]).astype(np.float32)
        q = ((torch.from_numpy(qvec).float().cuda() - self.qpos_mean)
             / self.qpos_std).unsqueeze(0)

        # 모델 추론 및 버퍼 저장
        with torch.inference_mode():
            out = self.model(q, img)  # shape: (1, chunk_size, dim)
            seq = out.squeeze(0)      # (chunk_size, dim)
            # t_idx행, 열 t_idx:t_idx+chunk_size에 저장
            self.action_buffer[self.t_idx, self.t_idx:self.t_idx + self.args.chunk_size] = seq
            # 앙상블 계산
            agg = temporal_aggregate(self.action_buffer, self.t_idx, self.args.chunk_size)
            action_norm = agg.squeeze(0)
            self.t_idx += 1

        # 역정규화
        action_t = action_norm * self.qpos_std + self.qpos_mean
        action = action_t.cpu().numpy()


        # --- 슬라이더 예측값에 따른 속도(v) 계산 ---
        slider_idx = len(self.joint_names) - 1
        slider_val = action[slider_idx]
        v = -1.0 if slider_val < 1e-4 else 0.01
        # ------------------------------------------

        # --- position 명령 생성 (7번째 제외) ---
        position_cmd = action[:len(self.joint_names)].tolist()
        # 슬라이더 position은 사용하지 않으므로 0 또는 무시될 값으로 설정
        position_cmd[slider_idx] = 0.0
        # -------------------------------------

        # --- velocity 명령 생성 (오직 7번째만) ---
        velocity_cmd = [0.0] * len(self.joint_names)
        velocity_cmd[slider_idx] = v
        # -------------------------------------

        # JointState 메시지에 position/velocity 모두 채워서 전송
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names
        js.position = position_cmd
        js.velocity = velocity_cmd
        js.effort = []  # 필요 없으면 빈 리스트
        self.joint_pub.publish(js)

        # cmd_vel 은 그대로
        tw = Twist()
        tw.linear.x  = float(action[-2])
        tw.angular.z = float(action[-1])
        self.twist_pub.publish(tw)

        self.get_logger().info(f"t={self.t_idx}, pos={position_cmd}, vel={velocity_cmd}")

def main():
    rclpy.init()
    args = parse_args()
    node = AIInferenceNode(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
