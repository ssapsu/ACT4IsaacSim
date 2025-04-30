#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# 센서 데이터용 QoS: best_effort, keep_last(1)
sensor_qos = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class DatasetRecorder(Node):
    def __init__(self):
        super().__init__("dataset_recorder")

        # 파라미터: 최상위 저장 경로
        self.declare_parameter(
            "output_dir", os.path.expanduser("~/Documents/ACT4IsaacSim/data")
        )
        self.root_dir = (
            self.get_parameter("output_dir").get_parameter_value().string_value
        )

        # episode counter
        self.episode = 0

        # 브릿지
        self.bridge = CvBridge()

        # 최신 Twist/JointState (필요시 사용)
        self.latest_twist = None
        self.latest_joints = None
        self.create_subscription(
            Twist, "/cmd_vel", self.cb_update_twist, qos_profile=sensor_qos
        )
        self.create_subscription(
            JointState, "/joint_states", self.cb_update_joints, qos_profile=sensor_qos
        )

        # Trigger 서비스로 snapshot 찍기
        self.srv = self.create_service(
            Trigger, "/start_episode", self.handle_start_episode
        )
        self.get_logger().info("✅ /start_episode Trigger service ready")

        # 토픽별 저장 디렉터리 매핑
        self.img_dirs = {
            "color": "camera_color",
            "depth": "camera_depth",
            "left": "camera_left_color",
            "labels": "camera_semantic_labels",
            "segm": "camera_semantic_segmentation",
        }
        self.json_dirs = {
            "twist": "cmd_vel",
            "joints": "joint_states",
        }

    def cb_update_twist(self, msg: Twist):
        self.latest_twist = msg

    def cb_update_joints(self, msg: JointState):
        self.latest_joints = msg

    def handle_start_episode(self, request, response):
        self.episode += 1
        ep_dir = os.path.join(self.root_dir, f"episode_{self.episode:03d}")
        # 서브폴더 생성
        for d in list(self.img_dirs.values()) + list(self.json_dirs.values()):
            os.makedirs(os.path.join(ep_dir, d), exist_ok=True)
        self.get_logger().info(f"📂 New episode directory: {ep_dir}")

        # 한 프레임씩 동기적으로 받아오기
        msgs = self._block_for_all_once(timeout=2.0)
        if msgs is None:
            response.success = False
            response.message = "Timeout waiting for all topics"
            return response

        # 타임스탬프 문자열
        header = msgs["color"].header
        ts = datetime.fromtimestamp(
            header.stamp.sec + header.stamp.nanosec * 1e-9
        ).strftime("%Y-%m-%d_%H-%M-%S-%f")

        # --- 1) color ---
        img = self.bridge.imgmsg_to_cv2(msgs["color"], "bgr8")
        cv2.imwrite(os.path.join(ep_dir, self.img_dirs["color"], f"{ts}.png"), img)
        # --- 2) depth ---
        arr = self.bridge.imgmsg_to_cv2(msgs["depth"], desired_encoding="passthrough")
        np.save(os.path.join(ep_dir, self.img_dirs["depth"], f"{ts}.npy"), arr)
        # --- 3) left color ---
        img = self.bridge.imgmsg_to_cv2(msgs["left"], "bgr8")
        cv2.imwrite(os.path.join(ep_dir, self.img_dirs["left"], f"{ts}.png"), img)
        # --- 4) semantic_labels ---
        arr = self.bridge.imgmsg_to_cv2(msgs["labels"], desired_encoding="passthrough")
        np.save(os.path.join(ep_dir, self.img_dirs["labels"], f"{ts}.npy"), arr)
        # --- 5) semantic_segmentation ---
        arr = self.bridge.imgmsg_to_cv2(msgs["segm"], desired_encoding="passthrough")
        np.save(os.path.join(ep_dir, self.img_dirs["segm"], f"{ts}.npy"), arr)

        # --- 6) cmd_vel → JSON ---
        if self.latest_twist:
            data = {
                "linear": {
                    "x": self.latest_twist.linear.x,
                    "y": self.latest_twist.linear.y,
                    "z": self.latest_twist.linear.z,
                },
                "angular": {
                    "x": self.latest_twist.angular.x,
                    "y": self.latest_twist.angular.y,
                    "z": self.latest_twist.angular.z,
                },
            }
            with open(
                os.path.join(ep_dir, self.json_dirs["twist"], f"{ts}.json"), "w"
            ) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        # --- 7) joint_states → JSON ---
        if self.latest_joints:
            data = {
                "name": self.latest_joints.name,
                "position": list(self.latest_joints.position),
                "velocity": list(self.latest_joints.velocity),
                "effort": list(self.latest_joints.effort),
            }
            with open(
                os.path.join(ep_dir, self.json_dirs["joints"], f"{ts}.json"), "w"
            ) as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        response.success = True
        response.message = f"Episode {self.episode:03d} recorded @ {ts}"
        self.get_logger().info(response.message)
        return response

    def _block_for_all_once(self, timeout=2.0):
        """
        '/camera/...' 5개 토픽에서 한 프레임씩 메시지를 블로킹으로 받아옵니다.
        타임아웃 내에 모두 수신하지 못하면 None 리턴.
        """
        got = {}

        def make_cb(key):
            def cb(msg):
                if key not in got:
                    got[key] = msg

            return cb

        # 일회성 구독자 생성
        subs = [
            self.create_subscription(
                Image, "/camera/color", make_cb("color"), qos_profile=sensor_qos
            ),
            self.create_subscription(
                Image, "/camera/depth", make_cb("depth"), qos_profile=sensor_qos
            ),
            self.create_subscription(
                Image, "/camera/left_color", make_cb("left"), qos_profile=sensor_qos
            ),
            self.create_subscription(
                Image,
                "/camera/semantic_labels",
                make_cb("labels"),
                qos_profile=sensor_qos,
            ),
            self.create_subscription(
                Image,
                "/camera/semantic_segmentation",
                make_cb("segm"),
                qos_profile=sensor_qos,
            ),
        ]

        # spin 일으켜서 메시지 수신 대기
        t0 = time.time()
        while len(got) < 5 and (time.time() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)

        # 일회성 구독 해제
        for s in subs:
            self.destroy_subscription(s)

        if len(got) < 5:
            self.get_logger().warn(f"⏱ Timeout: got {len(got)}/5 messages")
            return None
        return got


def main(args=None):
    rclpy.init(args=args)
    node = DatasetRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
