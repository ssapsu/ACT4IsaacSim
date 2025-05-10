#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT-policy용 HDF5 샘플링 레코더
- 4개 이미지 + odom + joint_states 동기 녹화
- /start_record, /stop_record 서비스로 에피소드 제어
- sample_rate (Hz) 파라미터로 저장 주기 지정
- stop_record 시점에 현재 에피소드 파일 자동 닫음
- 노드 재시작 시 기존 episode_{i}.hdf5 스캔하여 인덱스 이어쓰기
"""
import os
import re
import threading
from collections import deque

import cv2
import h5py
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger


class SampledSyncRecorder(Node):
    def __init__(self):
        super().__init__("sampled_sync_recorder")
        # parameters
        self.declare_parameter(
            "output_dir",
            os.path.expanduser("~/Documents/ACT4IsaacSim/dataset/isaac_sim_example"),
        )
        self.declare_parameter("cam_height", 480)
        self.declare_parameter("cam_width", 640)
        self.declare_parameter("sample_rate", 10.0)
        # resolve paths and prepare output dir
        base = self.get_parameter("output_dir").get_parameter_value().string_value
        self.root = os.path.expanduser(base)
        os.makedirs(self.root, exist_ok=True)
        self.cam_h = (
            self.get_parameter("cam_height").get_parameter_value().integer_value
        )
        self.cam_w = self.get_parameter("cam_width").get_parameter_value().integer_value
        self.sample_rate = (
            self.get_parameter("sample_rate").get_parameter_value().double_value
        )
        # determine starting episode index
        max_idx = -1
        for fname in os.listdir(self.root):
            m = re.match(r"^episode_(\d+)\.hdf5$", fname)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        self.episode_index = max_idx + 1
        # placeholders for recording
        self.h5 = None
        self.ep_group = None
        self.sync_count = 0
        self.ep_last_pos = None
        self.ep_cum_dist = 0.0
        # buffers
        self.bridge = CvBridge()
        self.img_buf = {"color": None, "depth": None, "left_color": None}
        self.odom_buf = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.joint_initialized = False
        self.state_dim = None
        # subscriptions
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Image,
            "/camera/color",
            lambda msg: self._img_cb("color", msg, "bgr8"),
            qos_profile=qos,
        )
        self.create_subscription(
            Image,
            "/camera/depth",
            lambda msg: self._img_cb("depth", msg, "passthrough"),
            qos_profile=qos,
        )
        self.create_subscription(
            Image,
            "/camera/left_color",
            lambda msg: self._img_cb("left_color", msg, "bgr8"),
            qos_profile=qos,
        )
        cbg = ReentrantCallbackGroup()
        self.create_subscription(
            Odometry, "/odom", self._cb_odom, 10, callback_group=cbg
        )
        self.create_subscription(
            JointState, "/joint_states", self._cb_joint, 10, callback_group=cbg
        )
        self.create_service(
            Trigger, "/start_record", self.handle_start, callback_group=cbg
        )
        self.create_service(
            Trigger, "/stop_record", self.handle_stop, callback_group=cbg
        )
        self.get_logger().info("Ready: /start_record, /stop_record")
        # timer
        self.create_timer(1.0 / self.sample_rate, self._cb_sample, callback_group=cbg)

    def _img_cb(self, key, msg, fmt):
        cv_img = self.bridge.imgmsg_to_cv2(msg, fmt)
        if key in ["color", "left_color"]:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.img_buf[key] = (msg.header.stamp, cv_img)

    def _cb_odom(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        self.odom_buf.append((ts, (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z)))

    def _cb_joint(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joint_buf.append((ts, list(msg.name), list(msg.position)))

    def handle_start(self, req, res):
        # open new hdf5
        out_path = os.path.join(self.root, f"episode_{self.episode_index}.hdf5")
        self.h5 = h5py.File(out_path, "w")
        self.h5.attrs["sim"] = True
        obs = self.h5.create_group("observations")
        imgs = obs.create_group("images")
        for key in self.img_buf:
            shape = (0, self.cam_h, self.cam_w, 3)
            imgs.create_dataset(
                key,
                shape=shape,
                maxshape=(None, self.cam_h, self.cam_w, 3),
                dtype="uint8",
                chunks=(1, self.cam_h, self.cam_w, 3),
            )
        obs.create_dataset(
            "qpos", shape=(0, 9), maxshape=(None, 9), dtype="float32", chunks=(1, 9)
        )
        # reset counters
        self.ep_group = obs
        self.sync_count = 0
        self.ep_last_pos = None
        self.ep_cum_dist = 0.0
        self.joint_initialized = False
        res.success = True
        res.message = "started"
        return res

    def handle_stop(self, req, res):
        if not self.h5:
            res.success = False
            res.message = "not recording"
            return res
        self.h5.close()
        self.episode_index += 1
        self.h5 = None
        res.success = True
        res.message = "stopped and saved"
        return res

    def _cb_sample(self):
        if not self.h5:
            return
        entry = self.img_buf["color"]
        if not entry:
            return
        stamp, _ = entry
        ts = stamp.sec + stamp.nanosec * 1e-9
        # sample odom & joint
        od = self._pop(self.odom_buf, ts)
        j = self._pop(self.joint_buf, ts)
        if od:
            od_ts, od_pos, od_twist = od
        else:
            od_pos = (0, 0, 0)
            od_twist = (0, 0, 0)
        if self.ep_last_pos is None:
            self.ep_last_pos = np.array(od_pos)
        dist = np.linalg.norm(np.array(od_pos) - self.ep_last_pos)
        self.ep_cum_dist += dist
        self.ep_last_pos = np.array(od_pos)
        if j:
            j_ts, names, pos_all = j
        else:
            pos_all = []
        # select joints 1-6 and Slider_1
        if pos_all and not self.joint_initialized:
            names_j = names
            idxs = [names_j.index(f"joint_{i}") for i in range(1, 7)] + [
                names_j.index("Slider_1")
            ]
            self.joint_idxs = idxs
            self.joint_initialized = True
        # build qpos vector
        if self.joint_initialized and pos_all:
            pos_sel = np.array(pos_all)[self.joint_idxs]
        else:
            pos_sel = np.zeros(7)
        lin_sp = np.linalg.norm(od_pos)
        ang_sp = np.linalg.norm(od_twist)
        qpos_vec = np.concatenate([pos_sel, [lin_sp, ang_sp]], axis=0)  # (9,)
        # images: take first camera only or stack? here color only
        _, img0 = self.img_buf["color"]
        img = cv2.resize(img0, (self.cam_w, self.cam_h), interpolation=cv2.INTER_AREA)
        # append
        obs = self.h5["observations"]
        ds_img = obs["images"]["color"]
        ds_img.resize(self.sync_count + 1, axis=0)
        ds_img[self.sync_count] = img
        ds_q = obs["qpos"]
        ds_q.resize(self.sync_count + 1, axis=0)
        ds_q[self.sync_count] = qpos_vec
        self.sync_count += 1

    def _pop(self, buf, ts, tol=0.05):
        if not buf:
            return None
        diffs = [abs(t - ts) for t, *_ in buf]
        i = int(np.argmin(diffs))
        if diffs[i] <= tol:
            item = buf[i]
            del buf[i]
            return item
        return None


def main():
    rclpy.init()
    node = SampledSyncRecorder()
    exe = MultiThreadedExecutor()
    exe.add_node(node)
    try:
        exe.spin()
    finally:
        exe.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
