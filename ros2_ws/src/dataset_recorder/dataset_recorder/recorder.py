#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT-policy용 HDF5 샘플링 레코더
- 4개 이미지 + semantic_labels + odom + joint_states 전부 placeholder 포함 동기 녹화
- /start_record, /stop_record 서비스로 에피소드 제어
- sample_rate (Hz) 파라미터로 저장 주기 지정
"""
import os
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
from std_msgs.msg import String
from std_srvs.srv import Trigger


class SampledSyncRecorder(Node):
    def __init__(self):
        super().__init__("sampled_sync_recorder")

        # parameters
        self.declare_parameter(
            "output_dir", os.path.expanduser("~/Documents/ACT4IsaacSim/data")
        )
        self.declare_parameter("cam_height", 480)
        self.declare_parameter("cam_width", 640)
        self.declare_parameter("sample_rate", 10.0)  # Hz

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

        # HDF5 and episode
        self.episode_index = 0
        self.h5 = None
        self.ep_group = None
        self.sync_count = 0
        self.ep_last_pos = None
        self.ep_cum_dist = 0.0

        # buffers
        self.bridge = CvBridge()
        self.label_buf = deque(maxlen=200)
        self.odom_buf = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.img_buf = {
            "color": None,
            "depth": None,
            "left_color": None,
            "semantic_segmentation": None,
        }
        self.joint_initialized = False
        self.state_dim = None
        self.joint_lock = threading.Lock()

        # QoS
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # image subscribers (latest buffer)
        self.create_subscription(
            Image,
            "/camera/color",
            lambda msg: self._img_cb("color", msg, "bgr8"),
            qos_profile=sensor_qos,
        )
        self.create_subscription(
            Image,
            "/camera/depth",
            lambda msg: self._img_cb("depth", msg, "passthrough"),
            qos_profile=sensor_qos,
        )
        self.create_subscription(
            Image,
            "/camera/left_color",
            lambda msg: self._img_cb("left_color", msg, "bgr8"),
            qos_profile=sensor_qos,
        )
        self.create_subscription(
            Image,
            "/camera/semantic_segmentation",
            lambda msg: self._img_cb("semantic_segmentation", msg, "passthrough"),
            qos_profile=sensor_qos,
        )

        # other topic buffers
        cbg = ReentrantCallbackGroup()
        self.create_subscription(
            String,
            "/camera/semantic_labels",
            self.cb_labels_buffer,
            10,
            callback_group=cbg,
        )
        self.create_subscription(
            Odometry, "/odom", self.cb_odom_buffer, 10, callback_group=cbg
        )
        self.create_subscription(
            JointState, "/joint_states", self.cb_joints_buffer, 10, callback_group=cbg
        )

        # services
        self.create_service(
            Trigger, "/start_record", self.handle_start, callback_group=cbg
        )
        self.create_service(
            Trigger, "/stop_record", self.handle_stop, callback_group=cbg
        )
        self.get_logger().info("Ready: /start_record, /stop_record")

        # sampling timer
        period = 1.0 / self.sample_rate
        self.create_timer(period, self.cb_sample, callback_group=cbg)

    def _img_cb(self, key, msg, fmt):
        cv_img = self.bridge.imgmsg_to_cv2(msg, fmt)
        self.img_buf[key] = (msg.header.stamp, cv_img)

    def cb_labels_buffer(self, msg):
        ts = self.get_clock().now().nanoseconds * 1e-9
        self.label_buf.append((ts, msg.data))

    def cb_odom_buffer(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_buf.append((ts, (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))

    def cb_joints_buffer(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joint_buf.append(
            (ts, list(msg.position), list(msg.velocity), list(msg.effort))
        )

    def handle_start(self, req, res):
        if not self.h5:
            path = os.path.join(self.root, "sampled_fullsync.hdf5")
            self.h5 = h5py.File(path, "a")
            self.h5.require_group("episodes")
        ep_root = self.h5["episodes"]
        name = f"episode_{self.episode_index}"
        if name in ep_root:
            del ep_root[name]
        self.ep_group = ep_root.create_group(name)

        # init subgroups
        imgs = self.ep_group.create_group("images")
        for key in self.img_buf:
            shape = (
                (0, self.cam_h, self.cam_w, 3)
                if key in ["color", "left_color"]
                else (0, self.cam_h, self.cam_w)
            )
            maxs = (None,) + shape[1:]
            dtype = "uint8" if key in ["color", "left_color"] else "float32"
            imgs.create_dataset(
                key, shape=shape, maxshape=maxs, dtype=dtype, chunks=(1,) + shape[1:]
            )

        lbl = self.ep_group.create_group("semantic_labels")
        lbl.create_dataset(
            "value",
            shape=(0,),
            dtype=h5py.string_dtype(),
            maxshape=(None,),
            chunks=(1,),
        )

        od = self.ep_group.create_group("odom")
        od.create_dataset(
            "stamp", shape=(0,), dtype="float64", maxshape=(None,), chunks=(1,)
        )
        od.create_dataset(
            "pos", shape=(0, 3), dtype="float64", maxshape=(None, 3), chunks=(1, 3)
        )
        od.create_dataset(
            "orient", shape=(0, 4), dtype="float64", maxshape=(None, 4), chunks=(1, 4)
        )
        od.create_dataset(
            "cum_dist", shape=(0,), dtype="float64", maxshape=(None,), chunks=(1,)
        )

        js = self.ep_group.create_group("joint_states")

        self.sync_count = 0
        self.ep_last_pos = None
        self.ep_cum_dist = 0.0
        self.joint_initialized = False

        res.success = True
        res.message = "started"
        return res

    def handle_stop(self, req, res):
        if self.h5 and self.ep_group:
            self.episode_index += 1
            self.ep_group = None
            res.success = True
            res.message = "stopped"
        else:
            res.success = False
            res.message = "not recording"
        return res

    def cb_sample(self):
        if not self.ep_group:
            return
        # get latest color timestamp
        entry = self.img_buf.get("color")
        if entry is None:
            return
        stamp, _ = entry
        ts = stamp.sec + stamp.nanosec * 1e-9
        self.get_logger().info(f"Sample @ {ts:.6f}")

        # prepare image arrays
        arrs = {}
        for key, entry in self.img_buf.items():
            if entry is None:
                img = (
                    np.zeros((self.cam_h, self.cam_w, 3), dtype="uint8")
                    if key in ["color", "left_color"]
                    else np.zeros((self.cam_h, self.cam_w), dtype="float32")
                )
            else:
                _, img0 = entry
                interp = (
                    cv2.INTER_AREA
                    if key in ["color", "left_color"]
                    else cv2.INTER_NEAREST
                )
                arr = cv2.resize(img0, (self.cam_w, self.cam_h), interpolation=interp)
                if key in ["color", "left_color"]:
                    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                img = arr
            arrs[key] = img

        # pop nearest buffers
        lbl = self._pop_nearest(self.label_buf, ts)
        od = self._pop_nearest(self.odom_buf, ts)
        jnt = self._pop_nearest(self.joint_buf, ts)

        # placeholders
        lbl_data = lbl or ""
        if od is None:
            od_ts, od_pos, od_ori = 0.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        else:
            od_ts, od_pos, od_ori = od
        if self.ep_last_pos is None:
            self.ep_last_pos = np.array(od_pos)
        d = np.linalg.norm(np.array(od_pos) - self.ep_last_pos)
        self.ep_cum_dist += d
        self.ep_last_pos = np.array(od_pos)

        if jnt is None and self.state_dim is not None:
            zero = [0.0] * self.state_dim
            j_ts, j_pos, j_vel, j_eff = 0.0, zero, zero, zero
        elif jnt is not None:
            j_ts, j_pos, j_vel, j_eff = jnt
        else:
            j_ts, j_pos, j_vel, j_eff = 0.0, [], [], []

        # lazy init joint_states
        js_grp = self.ep_group["joint_states"]
        if not self.joint_initialized and jnt:
            _, pos, vel, eff = jnt
            self.state_dim = len(pos)
            js_grp.create_dataset(
                "stamp", shape=(0,), dtype="float64", maxshape=(None,), chunks=(1,)
            )
            js_grp.create_dataset(
                "position",
                shape=(0, self.state_dim),
                dtype="float32",
                maxshape=(None, self.state_dim),
                chunks=(1, self.state_dim),
            )
            js_grp.create_dataset(
                "velocity",
                shape=(0, self.state_dim),
                dtype="float32",
                maxshape=(None, self.state_dim),
                chunks=(1, self.state_dim),
            )
            js_grp.create_dataset(
                "effort",
                shape=(0, self.state_dim),
                dtype="float32",
                maxshape=(None, self.state_dim),
                chunks=(1, self.state_dim),
            )
            self.joint_initialized = True

        # append at idx
        idx = self.sync_count
        imgs_grp = self.ep_group["images"]
        lbl_ds = self.ep_group["semantic_labels"]["value"]
        od_grp = self.ep_group["odom"]

        # images
        for key, arr in arrs.items():
            ds = imgs_grp[key]
            ds.resize(idx + 1, axis=0)
            ds[idx] = arr

        # semantic_labels
        lbl_ds.resize(idx + 1, axis=0)
        lbl_ds[idx] = lbl_data

        # odom
        od_grp["stamp"].resize(idx + 1, axis=0)
        od_grp["stamp"][idx] = od_ts
        od_grp["pos"].resize(idx + 1, axis=0)
        od_grp["pos"][idx] = od_pos
        od_grp["orient"].resize(idx + 1, axis=0)
        od_grp["orient"][idx] = od_ori
        od_grp["cum_dist"].resize(idx + 1, axis=0)
        od_grp["cum_dist"][idx] = self.ep_cum_dist

        # joint_states
        if self.state_dim is not None:
            js_grp["stamp"].resize(idx + 1, axis=0)
            js_grp["stamp"][idx] = j_ts
            js_grp["position"].resize(idx + 1, axis=0)
            js_grp["position"][idx] = j_pos
            js_grp["velocity"].resize(idx + 1, axis=0)
            js_grp["velocity"][idx] = j_vel
            js_grp["effort"].resize(idx + 1, axis=0)
            js_grp["effort"][idx] = j_eff

        self.sync_count += 1

    def _pop_nearest(self, buf, ts, tol=0.05):
        if not buf:
            return None
        diffs = [abs(item[0] - ts) for item in buf]
        idx = int(np.argmin(diffs))
        if diffs[idx] <= tol:
            item = buf[idx]
            del buf[idx]
            return item
        return None


def main():
    rclpy.init()
    node = SampledSyncRecorder()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
