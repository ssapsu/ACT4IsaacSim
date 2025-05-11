#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDF5 observations 데이터 개수 계산 스크립트
- SampledSyncRecorder/HDF5 포맷에 맞춰 'observations/images' 및 'observations/qpos' dataset entry 수 출력
"""
import os
import argparse
import h5py


def count_entries(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        # 경우에 따라 'root' 그룹 아래에 'observations'가 있을 수 있음
        root_grp = f.get('root', f)
        obs_grp = root_grp.get('observations', None)
        if obs_grp is None:
            print("No 'observations' group found.")
            return

        # 이미지 카메라별 엔트리 수
        imgs_grp = obs_grp.get('images', None)
        if imgs_grp is not None:
            print('=== Images ===')
            for cam in imgs_grp:
                ds = imgs_grp[cam]
                print(f"{cam}: {ds.shape[0]} frames")
        else:
            print("No 'images' group under 'observations'.")

        # qpos 엔트리 수
        qpos_ds = obs_grp.get('qpos', None)
        if qpos_ds is not None:
            print('\n=== QPOS ===')
            num_q = qpos_ds.shape[0]
            print(f"qpos entries: {num_q}")
            # 첫 번째 행 값 출력
            first_q = qpos_ds[0]
            print(f"first qpos row (frame 0): {first_q}")
        else:
            print("No 'qpos' dataset under 'observations'.")

            print("No 'qpos' dataset under 'observations'.")


def main():
    parser = argparse.ArgumentParser(description='Count HDF5 observation entries')
    parser.add_argument('hdf5_file', help='Path to the episode HDF5 file')
    args = parser.parse_args()

    if not os.path.exists(args.hdf5_file):
        print(f"File not found: {args.hdf5_file}")
        return

    count_entries(args.hdf5_file)


if __name__ == '__main__':
    main()
