#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACTPolicy 기반 ROS2 실시간 추론 노드
- argparse로 설정값(ckpt_dir, camera_names 등) 입력
- /camera, /odom, /joint_states 토픽에서 데이터 수집
- 설정된 주기(sample_rate)마다 모델 추론
- terminal 출력과 함께 joint_command, cmd_vel 토픽에 발행
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
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import sys

# 학습할 때 쓰는 policy.py 경로 추가
sys.path.append("/home/hyeonsu/Documents/ACT4IsaacSim")
from policy import ACTPolicy
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser("Custom ACT evaluator")
    # 필수 인자
    parser.add_argument("--ckpt_dir",
                        default="/home/hyeonsu/Documents/ACT4IsaacSim/ckpt")
    parser.add_argument("--ckpt_name", default="policy_best.ckpt")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_names", nargs="+",
                        default=["color", "left_color"])
    # ACT 하이퍼파라
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    # ROS2 노드 추가 파라미터
    parser.add_argument("--sample_rate", type=float, default=10.0)
    parser.add_argument("--stats_file", default="dataset_stats.pkl")
    return parser.parse_args()


def temporal_aggregate(
    all_time_actions: torch.Tensor,
    t: int,
    num_queries: int,
    k: float = 0.01
) -> torch.Tensor:
    """
    past actions(all_time_actions) 중에서 유효한(valid) 것들에
    지수 감쇠 가중치를 적용해 모아주는 함수
    """
    acts = all_time_actions[:, t]  # shape: (time, action_dim)
    mask = torch.all(acts != 0, dim=1)
    valid = acts[mask]
    if valid.numel() == 0:
        return torch.zeros(1, all_time_actions.size(-1),
                           device=all_time_actions.device)
    n = valid.size(0)
    weights = torch.exp(-k * torch.arange(n,
                                          device=valid.device,
                                          dtype=torch.float32))
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

        # 3) stats 로드 및 qpos_mean/std 구성
        stats_path = os.path.join(args.ckpt_dir, args.stats_file)
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

        # joint/vel 분리 통계로 저장했다면 합쳐서 사용
        if "joint_mean" in stats:
            jm = torch.tensor(stats["joint_mean"],
                              dtype=torch.float32).cuda()
            js = torch.tensor(stats["joint_std"],
                              dtype=torch.float32).cuda()
            vm = torch.tensor(stats["vel_mean"],
                              dtype=torch.float32).cuda()
            vs = torch.tensor(stats["vel_std"],
                              dtype=torch.float32).cuda()
            self.qpos_mean = torch.cat([jm, vm])
            self.qpos_std = torch.cat([js, vs])
        else:
            self.qpos_mean = torch.tensor(stats["qpos_mean"],
                                          dtype=torch.float32).cuda()
            self.qpos_std = torch.tensor(stats["qpos_std"],
                                         dtype=torch.float32).cuda()

        # 4) 퍼블리셔 생성
        self.joint_pub = self.create_publisher(JointState,
                                               'joint_command', 10)
        self.twist_pub = self.create_publisher(Twist,
                                               '/cmd_vel', 10)

        # 5) 브리지·버퍼 초기화
        self.bridge = CvBridge()
        self.img_buf = {cam: None for cam in args.camera_names}
        self.img_stamp = None
        self.odom_buf = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.joint_initialized = False

        # 6) chunk_size > 1 일 때 temporal buffer 준비
        if args.chunk_size > 1:
            T = 10000
            dim = self.qpos_mean.numel()
            self.action_buffer = torch.zeros(
                (T, T + args.chunk_size, dim), device="cuda")
            self.t_idx = 0

        # 7) 토픽 구독
        qos = 10
        for cam in args.camera_names:
            self.create_subscription(
                Image,
                f"/camera/{cam}",
                lambda msg, c=cam: self.img_cb(c, msg),
                qos
            )
        self.create_subscription(Odometry, "/odom", self.odom_cb, qos)
        self.create_subscription(JointState, "/joint_states",
                                 self.joint_cb, qos)

        # 8) 주기적 추론 타이머
        self.create_timer(1.0 / args.sample_rate, self.timer_cb)
        self.get_logger().info(
            f"Inference started: cams={args.camera_names} @ {args.sample_rate}Hz"
        )

    def img_cb(self, cam, msg):
        # ROS Image -> OpenCV BGR -> RGB
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.img_buf[cam] = (ts, rgb)
        self.img_stamp = ts

    def odom_cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        self.odom_buf.append(
            (ts, (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z))
        )

    def joint_cb(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joint_buf.append((ts, msg.name, msg.position))

    def _pop(self, buf, ts, tol=0.05):
        # 가장 가까운 timestamp 매칭 후 반환
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
        # 1) 모든 카메라 이미지가 준비될 때까지 대기
        if any(v is None for v in self.img_buf.values()):
            return

        ts = self.img_stamp

        # 2) 이미지 텐서로 변환
        cams = [
            rearrange(self.img_buf[c][1], "H W C -> C H W")
            for c in self.args.camera_names
        ]
        img = torch.from_numpy(np.stack(cams) / 255.0).float().cuda().unsqueeze(0)

        # 3) odom, joint 동기화해서 뽑기
        od = self._pop(self.odom_buf, ts)
        j  = self._pop(self.joint_buf, ts)

        # 4) joint_names 1회 초기화
        if j and not self.joint_initialized:
            _, names, poses = j
            idxs = [names.index(f"joint_{i}") for i in range(1, 7)]
            idxs += [names.index("Slider_1")]
            self.joint_idxs = idxs
            self.joint_names = [names[i] for i in idxs]
            self.joint_initialized = True

        # 5) qpos 벡터 구성
        if j and self.joint_initialized:
            _, _, poses = j
            pos_sel = np.array(poses)[self.joint_idxs]
        else:
            pos_sel = np.zeros(len(self.joint_names))

        lin_sp = np.linalg.norm(np.array(od[1]) if od else np.zeros(3))
        ang_sp = np.linalg.norm(np.array(od[2]) if od else np.zeros(3))

        qvec = np.concatenate([pos_sel, [lin_sp, ang_sp]]).astype(np.float32)
        q = (torch.from_numpy(qvec).cuda().float() -
             self.qpos_mean) / self.qpos_std
        q = q.unsqueeze(0)  # shape: (1, 9)

        # 6) 모델 추론 (normalized action)
        with torch.inference_mode():
            out = self.model(q, img)
            if self.args.chunk_size > 1:
                t = self.t_idx
                self.action_buffer[t, t:t + self.args.chunk_size] = out.squeeze(0)
                agg = temporal_aggregate(self.action_buffer,
                                         t, self.args.chunk_size)
                action_norm = agg.squeeze(0).cpu().numpy()
                self.t_idx += 1
            else:
                action_norm = out[0, 0].cpu().numpy()

        # 7) 역정규화 (unnormalize)
        #    action_norm 는 shape (9,)인 normalized qpos
        action_t = torch.from_numpy(action_norm).cuda().float()
        action_unnorm_t = action_t * self.qpos_std + self.qpos_mean
        action = action_unnorm_t.cpu().numpy()  # shape (9,)

        # 8) 퍼블리시: JointState
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.name = self.joint_names
        # 앞 len(joint_names)개는 joint position
        js_msg.position = action[:len(self.joint_names)].tolist()
        self.joint_pub.publish(js_msg)

        # 9) 퍼블리시: Twist (cmd_vel)
        tw = Twist()
        # 뒤 2개가 [lin_vel, ang_vel]
        tw.linear.x  = float(action[-2])
        tw.angular.z = float(action[-1])
        self.twist_pub.publish(tw)

        # 10) 로그
        self.get_logger().info(
            f"[Inference] joints={js_msg.position}, "
            f"cmd_vel=({tw.linear.x:.3f}, {tw.angular.z:.3f})"
        )



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
