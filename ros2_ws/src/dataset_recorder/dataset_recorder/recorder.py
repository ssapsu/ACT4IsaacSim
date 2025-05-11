#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT-policy용 HDF5 샘플링 레코더
- 3개 이미지(color, depth, left_color) + odom + joint_states 동기 녹화
- /start_record, /stop_record 서비스로 에피소드 제어
- sample_rate (Hz) 파라미터로 저장 주기 지정
- stop_record 시점에 현재 에피소드 파일 자동 닫음
- 노드 재시작 시 기존 episode_{i}.hdf5 스캔하여 인덱스 이어쓰기
- HDF5 I/O는 별도 쓰레드에서 처리하여 녹화 콜백 블로킹 방지
- message_filters ApproximateTimeSynchronizer로 이미지 동기화
"""
import os
import re
import threading
import time
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
from message_filters import Subscriber as MFSubscriber, ApproximateTimeSynchronizer


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
        base = self.get_parameter("output_dir").get_parameter_value().string_value
        self.root = os.path.expanduser(base)
        os.makedirs(self.root, exist_ok=True)
        self.cam_h = self.get_parameter("cam_height").get_parameter_value().integer_value
        self.cam_w = self.get_parameter("cam_width").get_parameter_value().integer_value
        self.sample_rate = self.get_parameter("sample_rate").get_parameter_value().double_value

        # determine starting episode index
        max_idx = -1
        for fname in os.listdir(self.root):
            m = re.match(r"^episode_(\d+)\.hdf5$", fname)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        self.episode_index = max_idx + 1

        # placeholders
        self.h5 = None
        self.sync_count = 0
        self.ep_last_pos = None
        self.ep_cum_dist = 0.0

        # buffers
        self.bridge = CvBridge()
        self.img_buf = {"color": None, "depth": None, "left_color": None}
        self.odom_buf = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.joint_initialized = False

        # writer queue & thread control
        self._write_queue = deque()
        self._write_lock = threading.Lock()
        self._writer_active = False
        self._writer_thread = None

        # callback group for non-image topics
        cbg = ReentrantCallbackGroup()

        # QoS for image topics
        qos_img = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # message_filters Subscribers for synchronized image callbacks
        sub_color = MFSubscriber(self, Image, '/camera/color', qos_profile=qos_img)
        sub_depth = MFSubscriber(self, Image, '/camera/depth', qos_profile=qos_img)
        sub_left  = MFSubscriber(self, Image, '/camera/left_color', qos_profile=qos_img)
        ats = ApproximateTimeSynchronizer(
            [sub_color, sub_depth, sub_left],
            queue_size=10,
            slop=0.05,
            allow_headerless=False
        )
        ats.registerCallback(self._imgs_cb)

        # remaining subscriptions and services
        self.create_subscription(
            Odometry, '/odom', self._cb_odom, 10, callback_group=cbg
        )
        self.create_subscription(
            JointState, '/joint_states', self._cb_joint, 10, callback_group=cbg
        )
        self.create_service(
            Trigger, '/start_record', self.handle_start, callback_group=cbg
        )
        self.create_service(
            Trigger, '/stop_record', self.handle_stop, callback_group=cbg
        )
        self.get_logger().info("Ready: /start_record, /stop_record")

        # timer for sampling
        self.create_timer(1.0 / self.sample_rate, self._cb_sample, callback_group=cbg)

    def _imgs_cb(self, color_msg, depth_msg, left_msg):
        # synchronized image callback
        ts = color_msg.header.stamp
        img_c = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        img_d = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        img_l = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
        img_c = cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        self.img_buf['color']      = (ts, img_c)
        self.img_buf['depth']      = (ts, img_d)
        self.img_buf['left_color'] = (ts, img_l)

    def _cb_odom(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        lin = msg.twist.twist.linear
        ang = msg.twist.twist.angular
        self.odom_buf.append((ts, (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z)))

    def _cb_joint(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.joint_buf.append((ts, list(msg.name), list(msg.position)))

    def handle_start(self, req, res):
        # open new HDF5
        out_path = os.path.join(self.root, f"episode_{self.episode_index}.hdf5")
        self.h5 = h5py.File(out_path, 'w')
        self.h5.attrs['sim'] = True
        obs = self.h5.create_group('observations')
        imgs_grp = obs.create_group('images')
        for key in self.img_buf:
            imgs_grp.create_dataset(
                key,
                shape=(0, self.cam_h, self.cam_w, 3),
                maxshape=(None, self.cam_h, self.cam_w, 3),
                dtype='uint8',
                chunks=(1, self.cam_h, self.cam_w, 3),
            )
        obs.create_dataset('qpos', shape=(0,9), maxshape=(None,9), dtype='float32', chunks=(1,9))
        # reset
        self.sync_count = 0
        self.ep_last_pos = None
        self.ep_cum_dist = 0.0
        self.joint_initialized = False
        # start writer thread
        self._writer_active = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
        res.success = True
        res.message = 'started'
        return res

    def handle_stop(self, req, res):
        if not self.h5:
            res.success = False
            res.message = 'not recording'
            return res
        self._writer_active = False
        self._writer_thread.join()
        self.h5.close()
        self.episode_index += 1
        self.h5 = None
        res.success = True
        res.message = 'stopped and saved'
        return res

    def _cb_sample(self):
        if not self.h5:
            return
        # 모든 이미지 채널 동기 처리 완료 여부 확인
        if any(self.img_buf[k] is None for k in ['color','depth','left_color']):
            return
        stamp, _ = self.img_buf['color']
        ts = stamp.sec + stamp.nanosec * 1e-9
        od = self._pop(self.odom_buf, ts)
        j  = self._pop(self.joint_buf, ts)
        if od:
            _, od_pos, od_twist = od
        else:
            od_pos, od_twist = (0,0,0),(0,0,0)
        if self.ep_last_pos is None:
            self.ep_last_pos = np.array(od_pos)
        dist = np.linalg.norm(np.array(od_pos)-self.ep_last_pos)
        self.ep_cum_dist += dist
        self.ep_last_pos = np.array(od_pos)
        if j:
            _, names, pos_all = j
        else:
            pos_all = []
        if pos_all and not self.joint_initialized:
            idxs = [names.index(f'joint_{i}') for i in range(1,7)] + [names.index('Slider_1')]
            self.joint_idxs = idxs
            self.joint_initialized = True
        if self.joint_initialized and pos_all:
            pos_sel = np.array(pos_all)[self.joint_idxs]
        else:
            pos_sel = np.zeros(7)
        lin_sp = np.linalg.norm(od_pos)
        ang_sp = np.linalg.norm(od_twist)
        qpos_vec = np.concatenate([pos_sel,[lin_sp,ang_sp]],axis=0)
        # 이미지 복사 및 resize
        imgs = {}
        for key, v in self.img_buf.items():
            _, cv_img = v
            if cv_img.ndim==2:
                cv_img = np.repeat(cv_img[:,:,None],3,axis=2)
            imgs[key] = cv2.resize(cv_img,(self.cam_w,self.cam_h),interpolation=cv2.INTER_AREA)
        # enqueue
        with self._write_lock:
            self._write_queue.append({'idx':self.sync_count,'imgs':imgs,'qpos':qpos_vec.copy()})
            self.sync_count += 1
        # clear buffers
        for k in self.img_buf:
            self.img_buf[k]=None

    def _writer_loop(self):
        while self._writer_active or self._write_queue:
            task=None
            with self._write_lock:
                if self._write_queue:
                    task=self._write_queue.popleft()
            if not task:
                time.sleep(0.005)
                continue
            idx=task['idx']
            imgs=task['imgs']
            qpos=task['qpos']
            obs=self.h5['observations']
            for key,img in imgs.items():
                ds=obs['images'][key]
                ds.resize(idx+1,axis=0)
                ds[idx]=img
            ds_q=obs['qpos']
            ds_q.resize(idx+1,axis=0)
            ds_q[idx]=qpos

    def _pop(self, buf, ts, tol=0.05):
        if not buf: return None
        diffs=[abs(t-ts) for t,*_ in buf]
        i=int(np.argmin(diffs))
        if diffs[i]<=tol:
            item=buf[i]
            del buf[i]
            return item
        return None


def main():
    rclpy.init()
    node=SampledSyncRecorder()
    exe=MultiThreadedExecutor(num_threads=4)
    exe.add_node(node)
    try:
        exe.spin()
    finally:
        exe.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    main()
