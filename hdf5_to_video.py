#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 observations 영상 생성 스크립트
- SampledSyncRecorder/HDF5 포맷에 맞춰 'observations/images' 및 'observations/qpos' 데이터를 영상으로 저장
- --camera 옵션으로 사용할 이미지 스트림 선택
- --overlay_qpos 옵션으로 각 프레임 위에 qpos 값 오버레이
"""
import os
import argparse
import h5py
import cv2
import numpy as np

def make_video(hdf5_path, camera_name, output_path, fps, overlay_qpos):
    # HDF5 파일 열기
    with h5py.File(hdf5_path, 'r') as f:
        root = f.get('root', f)
        obs = root.get('observations', None)
        if obs is None:
            raise RuntimeError("No 'observations' group in HDF5 file.")

        imgs_grp = obs.get('images', None)
        if imgs_grp is None or camera_name not in imgs_grp:
            raise RuntimeError(f"No images/{camera_name} dataset in HDF5.")
        img_ds = imgs_grp[camera_name]

        # qpos dataset
        qpos_ds = obs.get('qpos', None)
        if qpos_ds is None:
            raise RuntimeError("No 'qpos' dataset in observations.")

        num_frames = img_ds.shape[0]
        # 첫 프레임으로 해상도 확인
        frame0 = img_ds[0]
        if frame0.ndim == 2:
            height, width = frame0.shape
            channels = 1
        else:
            height, width, channels = frame0.shape

        # VideoWriter 준비 (BGR 포맷)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for idx in range(num_frames):
            img = img_ds[idx]
            # 그레이스케일 -> BGR
            if img.ndim == 2:
                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                # H, W, C 형식 assumed RGB
                frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if overlay_qpos:
                q = qpos_ds[idx]
                # qpos 벡터 문자열 생성
                txt = 'qpos:' + ','.join([f"{x:.2f}" for x in q.tolist()])
                # 표시 위치와 폰트 설정
                cv2.putText(
                    frame, txt,
                    (10, height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA
                )

            writer.write(frame)
        writer.release()


def main():
    parser = argparse.ArgumentParser(description='HDF5 to Video')
    parser.add_argument('hdf5_file', help='Path to HDF5 episode file')
    parser.add_argument('--camera', required=True,
                        help='Name of camera stream (e.g., color, left_color)')
    parser.add_argument('--output', required=True,
                        help='Output video file path (e.g., out.mp4)')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Frames per second for output video')
    parser.add_argument('--overlay_qpos', action='store_true',
                        help='Overlay qpos values on each frame')
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_file):
        print(f"Error: file not found {args.hdf5_file}")
        return

    try:
        make_video(args.hdf5_file, args.camera,
                   args.output, args.fps, args.overlay_qpos)
        print(f"Saved video to {args.output}")
    except Exception as e:
        print(f"Failed to create video: {e}")

if __name__ == '__main__':
    main()
