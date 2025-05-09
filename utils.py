import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader

import IPython

e = IPython.embed


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(
        self, episode_ids, dataset_dir, camera_names, norm_stats, action_offset
    ):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.action_offset = action_offset  # future frame offset
        self.is_sim = None
        # trigger initialization only if dataset non-empty
        if len(episode_ids) > 0:
            self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        ep_id = self.episode_ids[index]
        path = os.path.join(self.dataset_dir, f"episode_{ep_id}.hdf5")
        with h5py.File(path, "r") as f:
            self.is_sim = f.attrs.get("sim", True)
            total_len = f["observations/qpos"].shape[0]

            # ensure start allows future sequence
            max_start = max(0, total_len - self.action_offset - 1)
            start = np.random.randint(0, max_start + 1)

            # current qpos
            qpos = f["observations/qpos"][start]

            # future qpos sequence (next action_offset frames)
            seq = f["observations/qpos"][start + 1 : start + 1 + self.action_offset]
            seq_len = seq.shape[0]
            Dq = f["observations/qpos"].shape[1]
            padded = np.zeros((self.action_offset, Dq), dtype=seq.dtype)
            padded[:seq_len] = seq
            mask = np.ones(self.action_offset, dtype=bool)
            mask[:seq_len] = False

            # images at current step
            imgs = []
            for cam in self.camera_names:
                arr = f[f"observations/images/{cam}"][start]
                # ensure 3-channel format
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=2)
                imgs.append(arr)
            imgs = np.stack(imgs, axis=0)  # (cams, H, W, C)

        # to tensor & normalize
        img_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0

        q_t = torch.from_numpy(qpos).float()
        q_t = (q_t - torch.from_numpy(self.norm_stats["qpos_mean"])) / torch.from_numpy(
            self.norm_stats["qpos_std"]
        )

        fut_t = torch.from_numpy(padded).float()
        fut_t = (
            fut_t - torch.from_numpy(self.norm_stats["qpos_mean"])
        ) / torch.from_numpy(self.norm_stats["qpos_std"])

        m_t = torch.from_numpy(mask)

        return img_t, q_t, fut_t, m_t


def get_norm_stats(dataset_dir, episode_ids):
    all_q = []
    for ep_id in episode_ids:
        path = os.path.join(dataset_dir, f"episode_{ep_id}.hdf5")
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            all_q.append(torch.from_numpy(f["observations/qpos"][:]))
    if len(all_q) == 0:
        raise RuntimeError("No episodes found for normalization.")
    all_q = torch.cat(all_q, dim=0)

    q_mean = all_q.mean(dim=0)
    q_std = all_q.std(dim=0).clamp(min=1e-2)

    return {"qpos_mean": q_mean.numpy(), "qpos_std": q_std.numpy()}


def load_data(
    dataset_dir,
    num_episodes,
    camera_names,
    batch_size_train,
    batch_size_val,
    action_offset,
):
    # gather existing episode IDs
    all_ids = [
        ep_id
        for ep_id in range(num_episodes)
        if os.path.exists(os.path.join(dataset_dir, f"episode_{ep_id}.hdf5"))
    ]
    if not all_ids:
        raise RuntimeError(f"No episode files found in {dataset_dir}")

        # split into train/val: ensure at least one train and one val if possible
    ids = np.array(all_ids)
    np.random.shuffle(ids)
    if len(ids) < 2:
        # if only one episode, use it for both train and val
        train_ids = ids.tolist()
        val_ids = ids.tolist()
    else:
        split = int(0.8 * len(ids))
        split = max(1, split)
        train_ids = ids[:split].tolist()
        val_ids = ids[split:].tolist()

    # compute normalization on existing episodes on existing episodes
    stats = get_norm_stats(dataset_dir, train_ids + val_ids)

    # datasets and loaders
    train_ds = EpisodicDataset(
        train_ids, dataset_dir, camera_names, stats, action_offset
    )
    val_ds = EpisodicDataset(val_ids, dataset_dir, camera_names, stats, action_offset)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    # use sequential sampler for empty or non-empty val set
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    return train_loader, val_loader, stats, train_ds.is_sim


def compute_dict_mean(epoch_dicts):
    result = {}
    for k in epoch_dicts[0].keys():
        vals = [d[k] for d in epoch_dicts]
        result[k] = torch.stack(vals, dim=0).mean(dim=0)
    return result


def detach_dict(d):
    return {k: v.detach() for k, v in d.items()}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
