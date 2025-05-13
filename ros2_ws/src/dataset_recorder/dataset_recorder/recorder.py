#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACT-policy용 HDF5 샘플링 레코더
- 4개 이미지(color, left_color, CCTV_color, arm_color) + odom + joint_states 동기 녹화
- /start_record, /stop_record 서비스로 에피소드 제어
- sample_rate (Hz) 파라미터로 저장 주기 지정
- stop_record 시점에 현재 에피소드 파일 자동 닫음
- 노드 재시작 시 기존 episode_{i}.hdf5 스캔하여 인덱스 이어쓰기
- HDF5 I/O는 ThreadPoolExecutor를 이용해 비동기 처리
- message_filters ApproximateTimeSynchronizer로 이미지 동기화
"""
import os
import re
from collections import deque
import cv2
import h5py
import numpy as np
import rclpy
from concurrent.futures import ThreadPoolExecutor
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, JointState
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger
from message_filters import Subscriber as MFSubscriber, ApproximateTimeSynchronizer


class SampledSyncRecorder(Node):
    def __init__(self):
        super().__init__('sampled_sync_recorder')
        # parameters
        self.declare_parameter('output_dir', os.path.expanduser('~/Documents/ACT4IsaacSim/dataset/isaac_sim_example'))
        self.declare_parameter('cam_height', 480)
        self.declare_parameter('cam_width', 640)
        self.declare_parameter('sample_rate', 30.0)
        base = self.get_parameter('output_dir').get_parameter_value().string_value
        self.root = os.path.expanduser(base)
        os.makedirs(self.root, exist_ok=True)
        self.cam_h = self.get_parameter('cam_height').get_parameter_value().integer_value
        self.cam_w = self.get_parameter('cam_width').get_parameter_value().integer_value
        self.sample_rate = self.get_parameter('sample_rate').get_parameter_value().double_value

        # determine starting episode index
        max_idx = -1
        for fname in os.listdir(self.root):
            m = re.match(r'^episode_(\d+)\.hdf5$', fname)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        self.episode_index = max_idx + 1

        # placeholders and buffers
        self.h5 = None
        self.sync_count = 0
        self.ep_last_pos = None
        self.ep_cum_dist = 0.0
        self.bridge = CvBridge()
        self.img_buf = {'color': None, 'left_color': None, 'CCTV_color': None, 'arm_color': None}
        self.odom_buf = deque(maxlen=200)
        self.joint_buf = deque(maxlen=200)
        self.joint_initialized = False

        # thread pool for async writes
        self._writer_executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)

        # callback group
        cbg = ReentrantCallbackGroup()
        # QoS
        qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         durability=QoSDurabilityPolicy.VOLATILE,
                         history=QoSHistoryPolicy.KEEP_LAST,
                         depth=1)
        # image subscribers
        sub_color = MFSubscriber(self, Image, '/camera/color', qos_profile=qos)
        sub_left = MFSubscriber(self, Image, '/camera/left_color', qos_profile=qos)
        sub_cctv = MFSubscriber(self, Image, '/camera/CCTV_color', qos_profile=qos)
        sub_arm = MFSubscriber(self, Image, '/camera/arm_color', qos_profile=qos)
        ats = ApproximateTimeSynchronizer([sub_color, sub_left, sub_cctv, sub_arm], queue_size=10, slop=0.05, allow_headerless=False)
        ats.registerCallback(self._imgs_cb)

        # other subscriptions and services
        self.create_subscription(Odometry, '/odom', self._cb_odom, 10, callback_group=cbg)
        self.create_subscription(JointState, '/joint_states', self._cb_joint, 10, callback_group=cbg)
        self.create_service(Trigger, '/start_record', self.handle_start, callback_group=cbg)
        self.create_service(Trigger, '/stop_record', self.handle_stop, callback_group=cbg)
        self.get_logger().info('Ready: call /start_record and /stop_record')

        # sampling timer
        self.create_timer(1.0/self.sample_rate, self._cb_sample, callback_group=cbg)

    def _imgs_cb(self, color_msg, left_msg, cctv_msg, arm_msg):
        ts = color_msg.header.stamp
        img_c = cv2.cvtColor(self.bridge.imgmsg_to_cv2(color_msg, 'bgr8'), cv2.COLOR_BGR2RGB)
        img_l = cv2.cvtColor(self.bridge.imgmsg_to_cv2(left_msg, 'bgr8'), cv2.COLOR_BGR2RGB)
        img_v = cv2.cvtColor(self.bridge.imgmsg_to_cv2(cctv_msg, 'bgr8'), cv2.COLOR_BGR2RGB)
        img_a = cv2.cvtColor(self.bridge.imgmsg_to_cv2(arm_msg, 'bgr8'), cv2.COLOR_BGR2RGB)
        self.img_buf['color'] = (ts, img_c)
        self.img_buf['left_color'] = (ts, img_l)
        self.img_buf['CCTV_color'] = (ts, img_v)
        self.img_buf['arm_color'] = (ts, img_a)

    def _cb_odom(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        lin = msg.twist.twist.linear; ang = msg.twist.twist.angular
        self.odom_buf.append((ts, (lin.x, lin.y, lin.z), (ang.x, ang.y, ang.z)))

    def _cb_joint(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
        self.joint_buf.append((ts, list(msg.name), list(msg.position)))

    def handle_start(self, req, res):
        path = os.path.join(self.root, f'episode_{self.episode_index}.hdf5')
        self.h5 = h5py.File(path, 'w')
        self.h5.attrs['sim'] = True
        obs = self.h5.create_group('observations')
        imgs_grp = obs.create_group('images')
        for key in self.img_buf:
            imgs_grp.create_dataset(key, shape=(0, self.cam_h, self.cam_w, 3), maxshape=(None, self.cam_h, self.cam_w, 3), dtype='uint8', chunks=(1,self.cam_h,self.cam_w,3))
        obs.create_dataset('qpos', shape=(0,9), maxshape=(None,9), dtype='float32', chunks=(1,9))
        self.sync_count = 0; self.ep_last_pos = None; self.ep_cum_dist = 0.0; self.joint_initialized = False
        res.success = True; res.message = 'started'
        return res

    def handle_stop(self, req, res):
        if not self.h5:
            res.success = False; res.message = 'not recording'; return res
        self._writer_executor.shutdown(wait=True)
        self.h5.close()
        self.episode_index += 1; self.h5 = None
        res.success = True; res.message = 'stopped and saved'
        return res

    def _cb_sample(self):
        if not self.h5: return
        if any(self.img_buf[k] is None for k in self.img_buf): return
        ts = self.img_buf['color'][0].sec + self.img_buf['color'][0].nanosec*1e-9
        od = self._pop(self.odom_buf, ts); j = self._pop(self.joint_buf, ts)
        pos, twist = (od[1], od[2]) if od else ((0,0,0),(0,0,0))
        if self.ep_last_pos is None: self.ep_last_pos = np.array(pos)
        dist = np.linalg.norm(np.array(pos)-self.ep_last_pos)
        self.ep_cum_dist += dist; self.ep_last_pos = np.array(pos)
        if j and not self.joint_initialized:
            _, names, all_pos = j
            idxs = [names.index(f'joint_{i}') for i in range(1,7)] + [names.index('Slider_1')]
            self.joint_idxs = idxs; self.joint_initialized = True
        qpos = np.zeros(9)
        if self.joint_initialized and j:
            _, _, all_pos = j
            sel = np.array(all_pos)[self.joint_idxs]
            lin_sp = np.linalg.norm(pos); ang_sp = np.linalg.norm(twist)
            qpos = np.concatenate([sel, [lin_sp, ang_sp]])
        imgs = {}
        for k,(stamp,img) in self.img_buf.items():
            if img.ndim==2: img = np.repeat(img[:,:,None],3,axis=2)
            imgs[k] = cv2.resize(img, (self.cam_w,self.cam_h), interpolation=cv2.INTER_AREA)
        idx = self.sync_count; self.sync_count += 1
        self._writer_executor.submit(self._write_once, idx, imgs, qpos)
        for k in self.img_buf: self.img_buf[k] = None

    def _write_once(self, idx, imgs, qpos):
        obs = self.h5['observations']
        for k,img in imgs.items():
            ds = obs['images'][k]; ds.resize(idx+1, axis=0); ds[idx] = img
        dsq = obs['qpos']; dsq.resize(idx+1, axis=0); dsq[idx] = qpos

    def _pop(self, buf, ts, tol=0.05):
        if not buf: return None
        diffs = [abs(t-ts) for t, *rest in buf]; i = int(np.argmin(diffs))
        if diffs[i] <= tol:
            item = buf[i]; del buf[i]; return item
        return None


def main():
    rclpy.init()
    node = SampledSyncRecorder()
    exe = MultiThreadedExecutor(num_threads=8)
    exe.add_node(node)
    try:
        exe.spin()
    finally:
        exe.shutdown(); node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
