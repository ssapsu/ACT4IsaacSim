from .detr_vae import build as build_vae


def build_ACT_model(args):
    return build_vae(args)
