#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT-policy용 HDF5 샘플링 레코더
- 4개 이미지 + odom + joint_states 전부 placeholder 포함 동기 녹화
- /start_record, /stop_record 서비스로 에피소드 제어
- sample_rate (Hz) 파라미터로 저장 주기 지정
- stop_record 시점에 별도 HDF5 파일로 export (EpisodicDataset 호환 포맷)
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
        self.odom_buf = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.img_buf = {"color": None, "depth": None, "left_color": None}
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

        # image subscribers
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

        # other topic buffers
        cbg = ReentrantCallbackGroup()
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
        if key in ["color", "left_color"]:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        self.img_buf[key] = (msg.header.stamp, cv_img)

    def cb_odom_buffer(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        self.odom_buf.append((ts, (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z)))

    def cb_joints_buffer(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joint_buf.append((ts, msg.name, list(msg.position)))

    def handle_start(self, req, res):
        # open fullsync file
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

        od = self.ep_group.create_group("odom")
        od.create_dataset(
            "stamp", shape=(0,), dtype="float64", maxshape=(None,), chunks=(1,)
        )
        od.create_dataset(
            "pos", shape=(0, 3), dtype="float64", maxshape=(None, 3), chunks=(1, 3)
        )
        od.create_dataset(
            "orient", shape=(0, 3), dtype="float64", maxshape=(None, 3), chunks=(1, 3)
        )
        od.create_dataset(
            "cum_dist", shape=(0,), dtype="float64", maxshape=(None,), chunks=(1,)
        )

        js = self.ep_group.create_group("joint_states")
        self.joint_initialized = False

        # reset counters
        self.sync_count = 0
        self.ep_last_pos = None
        self.ep_cum_dist = 0.0
        res.success = True
        res.message = "started"
        return res

    def handle_stop(self, req, res):
        if not self.ep_group:
            res.success = False
            res.message = "not recording"
            return res

        # export this episode
        self._export_episode(self.episode_index)

        # cleanup for next episode
        self.ep_group = None
        self.odom_buf.clear()
        self.joint_buf.clear()
        self.img_buf = {k: None for k in self.img_buf}
        self.episode_index += 1

        res.success = True
        res.message = "stopped and exported"
        return res

    def cb_sample(self):
        if not self.ep_group:
            return
        # timestamp
        entry = self.img_buf.get("color")
        if entry is None:
            return
        stamp, _ = entry
        ts = stamp.sec + stamp.nanosec * 1e-9
        self.get_logger().info(f"Sample @ {ts:.6f}")

        # collect images
        arrs = {}
        for key, entry in self.img_buf.items():
            if entry is None:
                img = np.zeros((self.cam_h, self.cam_w, 3), dtype="uint8")
            else:
                _, img0 = entry
                img = cv2.resize(
                    img0, (self.cam_w, self.cam_h), interpolation=cv2.INTER_AREA
                )
            arrs[key] = img

        # pop nearest odom & joint
        od = self._pop_nearest(self.odom_buf, ts)
        jnt = self._pop_nearest(self.joint_buf, ts)

        # placeholders / compute cum_dist
        if od:
            od_ts, od_pos, od_twist = od
        else:
            od_ts, od_pos, od_twist = 0.0, (0, 0, 0), (0, 0, 0)
        if self.ep_last_pos is None:
            self.ep_last_pos = np.array(od_pos)
        d = np.linalg.norm(np.array(od_pos) - self.ep_last_pos)
        self.ep_cum_dist += d
        self.ep_last_pos = np.array(od_pos)

        if jnt:
            j_ts, names, pos_all = jnt
        else:
            j_ts, names, pos_all = 0.0, [], []

        # lazy init joint_states internal fullsync
        js_grp = self.ep_group["joint_states"]
        if not self.joint_initialized and jnt:
            self.state_dim = len(pos_all)
            js_grp.create_dataset(
                "stamp", shape=(0,), dtype="float64", maxshape=(None,), chunks=(1,)
            )
            js_grp.create_dataset(
                "name", data=np.array(names, dtype="S"), dtype=h5py.string_dtype()
            )
            js_grp.create_dataset(
                "position",
                shape=(0, self.state_dim),
                dtype="float32",
                maxshape=(None, self.state_dim),
                chunks=(1, self.state_dim),
            )
            self.joint_initialized = True

        idx = self.sync_count
        # append images
        imgs_grp = self.ep_group["images"]
        for key, arr in arrs.items():
            ds = imgs_grp[key]
            ds.resize(idx + 1, axis=0)
            ds[idx] = arr

        # append odom
        od_grp = self.ep_group["odom"]
        for field, val in [
            ("stamp", od_ts),
            ("pos", od_pos),
            ("orient", od_twist),
            ("cum_dist", self.ep_cum_dist),
        ]:
            ds = od_grp[field]
            ds.resize(idx + 1, axis=0)
            ds[idx] = val

        # joint_states
        if self.joint_initialized:
            js_grp["stamp"].resize(idx + 1, axis=0)
            js_grp["stamp"][idx] = j_ts
            js_grp["position"].resize(idx + 1, axis=0)
            js_grp["position"][idx] = pos_all

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

    def _export_episode(self, ep_idx):
        src = self.h5["episodes"][f"episode_{ep_idx}"]
        out_path = os.path.join(self.root, f"episode_{ep_idx}.hdf5")
        # --- prepare joint subset and odom speeds ---
        names = [n.decode() for n in src["joint_states"]["name"][:]]
        pos_all = src["joint_states"]["position"][:]
        sel = [f"joint_{i}" for i in range(1, 7)] + ["Slider_1"]
        idxs = [names.index(s) for s in sel]
        pos_sel = pos_all[:, idxs]  # (T,7)
        lin = src["odom"]["pos"][
            :
        ]  # misuse pos as linear? assume we stored twist in orient
        ang = src["odom"]["orient"][:]
        lin_speed = np.linalg.norm(lin, axis=1, keepdims=True)
        ang_speed = np.linalg.norm(ang, axis=1, keepdims=True)
        final_qpos = np.concatenate([pos_sel, lin_speed, ang_speed], axis=1)

        # --- write new file ---
        with h5py.File(out_path, "w") as dst:
            dst.attrs["sim"] = True
            obs = dst.create_group("observations")
            imgs = obs.create_group("images")
            for key in ["color", "depth", "left_color"]:
                data = src["images"][key][:]
                imgs.create_dataset(
                    key,
                    data=data,
                    dtype=data.dtype,
                    chunks=(1,) + data.shape[1:],
                    compression="gzip",
                    compression_opts=2,
                )
            obs.create_dataset("qpos", data=final_qpos.astype("float32"))
        self.get_logger().info(f"Exported episode to {out_path}")


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
