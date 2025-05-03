import h5py


def count_episode_entries(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        episodes = f.get("episodes", {})
        if not episodes:
            print("No 'episodes' group found.")
            return
        for ep_name in episodes:
            ep = episodes[ep_name]
            print(f"Episode: {ep_name}")
            # Images
            imgs = ep.get("images", {})
            for key in imgs:
                count = imgs[key].shape[0]
                print(f"  images/{key}: {count}")
            # Semantic labels
            lbl_count = ep["semantic_labels"]["value"].shape[0]
            print(f"  semantic_labels: {lbl_count}")
            # Odometry
            od_count = ep["odom"]["stamp"].shape[0]
            print(f"  odom (stamp entries): {od_count}")
            # Joint states
            js_count = ep["joint_states"]["stamp"].shape[0]
            print(f"  joint_states (stamp entries): {js_count}")
            print()


if __name__ == "__main__":
    # HDF5 파일 경로를 실제 경로로 수정하세요.
    hdf5_file = "./data/full_sync_dataset.hdf5"
    count_episode_entries(hdf5_file)
