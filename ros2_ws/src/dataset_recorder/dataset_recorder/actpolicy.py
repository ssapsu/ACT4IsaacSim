#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACTPolicy 기반 ROS2 실시간 추론 노드
- argparse로 설정값(ckpt_dir, camera_names 등) 입력
- /camera, /odom, /joint_states 토픽에서 데이터 수집
- 설정된 주기(sample_rate)마다 모델 추론
- 타임 앙상블(과거 chunk_size 히스토리 가중 평균)으로 단일 액션 발행
- joint_command, cmd_vel 토픽에 발행
- tabulate 라이브러리로 표 형태 로깅
- cmd_vel 예측값이 표준편차 이내일 경우 (평균 언저리) 0,0 전송
"""
import os
import sys
import argparse
import pickle
import time
from collections import deque

import cv2
import torch
import numpy as np
from einops import rearrange
from tabulate import tabulate  # pip install tabulate

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
    parser.add_argument("--camera_names", nargs="+", default=["color", "left_color", "CCTV_color", "arm_color"])
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=4096)
    parser.add_argument("--sample_rate", type=float, default=20.0)
    parser.add_argument("--stats_file", default="dataset_stats.pkl")
    return parser.parse_args()


def temporal_aggregate(all_time_actions: torch.Tensor, t: int, k: float = 0.01) -> torch.Tensor:
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
            "enc_layers": 8,
            "dec_layers": 8,
            "nheads": 16,
            "camera_names": args.camera_names,
        }
        self.model = ACTPolicy(policy_cfg)
        # self.model.load_state_dict(torch.load(ckpt_path))
        state_dict = torch.load(ckpt_path)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            self.get_logger().warn(f"Unexpected keys in state_dict: {unexpected}")
        if missing:
            self.get_logger().warn(f"Missing keys in state_dict: {missing}")
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

        # cmd_vel 필터용 임계치 설정 (속도에 대한 표준편차)
        # qpos_std 구조: [..., vel_std_linear, vel_std_angular]
        self.threshold_lin = float(self.qpos_std[-2].item())
        self.threshold_ang = float(self.qpos_std[-1].item())

        # 퍼블리셔 및 구독
        self.joint_pub = self.create_publisher(JointState, 'joint_command', 10)
        self.twist_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.last_cmd = None
        self.create_subscription(Twist, '/cmd_vel', self._cmd_cb, 10)

        # CV Bridge 및 버퍼
        self.bridge = CvBridge()
        self.img_buf   = {c: None for c in args.camera_names}
        self.img_stamp = None
        self.odom_buf  = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.joint_initialized = False

        # 앙상블 버퍼
        T = 10000
        dim = self.qpos_mean.numel()
        self.action_buffer = torch.zeros((T, T + args.chunk_size, dim), device="cuda")
        self.t_idx = 0

        # 토픽 구독
        qos = 10
        for cam in args.camera_names:
            self.create_subscription(
                Image, f"/camera/{cam}",
                lambda msg, c=cam: self.img_cb(c, msg),
                qos
            )
        self.create_subscription(Odometry, "/odom", self.odom_cb, qos)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, qos)

        # 타이머
        interval = 1.0 / args.sample_rate
        self.create_timer(interval, self.timer_cb)
        self.get_logger().info(f"Inference started @ {args.sample_rate}Hz")

    def _cmd_cb(self, msg: Twist):
        self.last_cmd = msg

    def img_cb(self, cam, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if cam == 'depth':
            depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            dmin, dmax = np.nanmin(depth), np.nanmax(depth)
            norm = (depth - dmin) / (dmax - dmin + 1e-6)
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
        self.odom_buf.append((ts, (lin.x,lin.y,lin.z), (ang.x,ang.y,ang.z)))

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
        if any(v is None for v in self.img_buf.values()):
            return
        ts = self.img_stamp
        cams = [rearrange(self.img_buf[c][1], "H W C -> C H W") for c in self.args.camera_names]
        img = torch.from_numpy(np.stack(cams)/255.0).float().cuda().unsqueeze(0)

        od = self._pop(self.odom_buf, ts)
        j  = self._pop(self.joint_buf, ts)
        if j and not self.joint_initialized:
            _, names, poses = j
            idxs = [names.index(f"joint_{i}") for i in range(1,7)] + [names.index("Slider_1")]
            self.joint_idxs = idxs
            self.joint_names = [names[i] for i in idxs]
            self.joint_initialized = True

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

        with torch.inference_mode():
            out = self.model(q, img)
            seq = out.squeeze(0)
            self.action_buffer[self.t_idx, self.t_idx:self.t_idx + self.args.chunk_size] = seq
            agg = temporal_aggregate(self.action_buffer, self.t_idx)
            action_norm = agg.squeeze(0)
            self.t_idx += 1

        action_t = action_norm * self.qpos_std + self.qpos_mean
        action = action_t.cpu().numpy()

        slider_idx = len(self.joint_names) - 1
        slider_val = action[slider_idx]
        v = -1.0 if slider_val < 1e-4 else 0.01

        position_cmd = action[:len(self.joint_names)].tolist()
        position_cmd[slider_idx] = 0.0
        velocity_cmd = [0.0] * len(self.joint_names)
        velocity_cmd[slider_idx] = v

        # cmd_vel 예측값이 표준편차 이내면 평균 언저리로 보고 0,0 전송
        pred_lin = float(action[-2])
        pred_ang = float(action[-1])
        if abs(pred_lin) < self.threshold_lin and abs(pred_ang) < self.threshold_ang:
            pred_lin, pred_ang = 0.0, 0.0

        # JointState 퍼블리시
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name     = self.joint_names
        js.position = position_cmd
        js.velocity = velocity_cmd
        js.effort   = []
        self.joint_pub.publish(js)

        # Twist 퍼블리시
        tw = Twist()
        tw.linear.x  = pred_lin
        tw.angular.z = pred_ang
        self.twist_pub.publish(tw)

        # tabulate로 표 출력
        headers = ["Robot Arm (6)", "Gripper", "PredLin", "PredAng", "ActLin/ActAng"]
        arm_str = ", ".join(f"{p:.4f}" for p in position_cmd[:6])
        gripper_val = velocity_cmd[slider_idx]
        act_lin = self.last_cmd.linear.x if self.last_cmd else float('nan')
        act_ang = self.last_cmd.angular.z if self.last_cmd else float('nan')
        row = [
            arm_str,
            f"{gripper_val:.4f}",
            f"{pred_lin:.4f}",
            f"{pred_ang:.4f}",
            f"{act_lin:.4f}/{act_ang:.4f}"
        ]
        table = tabulate([row], headers, tablefmt="grid")
        self.get_logger().info("\n" + table)


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
