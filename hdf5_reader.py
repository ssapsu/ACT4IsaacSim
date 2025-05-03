import h5py
import numpy as np
import cv2
import os


def read_fullsync_dataset(hdf5_path, output_dir="./"):
    with h5py.File(hdf5_path, "r") as f:
        # Episodes 그룹 확인
        episodes = f.get("episodes", {})
        print("Available episodes:", list(episodes.keys()), "\n")

        for ep_name, ep_grp in episodes.items():
            print(f"--- Reading {ep_name} ---")
            ep_out = os.path.join(output_dir, ep_name)
            os.makedirs(ep_out, exist_ok=True)

            # 1) 이미지 데이터
            imgs = ep_grp["images"]
            for key in imgs:
                ds = imgs[key]
                print(f"Image '{key}': shape={ds.shape}, dtype={ds.dtype}")
                if ds.ndim == 4 and ds.dtype == np.uint8 and ds.shape[0] > 0:
                    img = ds[0]
                    path = os.path.join(ep_out, f"{key}_frame0.png")
                    cv2.imwrite(path, img)
                    print(f"  -> Saved first frame to '{path}'")
            print()

            # 2) semantic_labels
            labels = ep_grp["semantic_labels"]["value"][:]
            print("Semantic labels:", labels)
            print()

            # 3) odometry
            od = ep_grp["odom"]
            stamps = od["stamp"][:]
            poss = od["pos"][:]
            orients = od["orient"][:]
            cumd = od["cum_dist"][:]
            print("odom:")
            print("  stamps:", stamps.shape)
            print("  positions:", poss.shape)
            print("  orientations:", orients.shape)
            print("  cumulative distances:", cumd.shape)
            if stamps.size > 0:
                print(
                    "  first odom ->",
                    stamps[0],
                    poss[0],
                    orients[0],
                    "cum_dist:",
                    cumd[0],
                )
            print()

            # 4) joint_states
            js = ep_grp["joint_states"]
            print("joint_states:")
            for field in ["stamp", "position", "velocity", "effort"]:
                data = js[field][:]
                print(f"  {field}: shape={data.shape}")
            if js["stamp"].size > 0:
                print("  first joint state ->", js["stamp"][0], js["position"][0])
            print("\n")


if __name__ == "__main__":
    # HDF5 파일 경로와 출력 디렉토리를 설정하세요.
    hdf5_file = "./data/full_sync_dataset.hdf5"
    output_dir = "./output_frames"
    read_fullsync_dataset(hdf5_file, output_dir)

    print("Done reading full sync dataset.")
