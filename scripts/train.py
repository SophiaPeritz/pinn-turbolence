#!/usr/bin/env python3
"""Script CLI per lanciare il training con un file di configurazione YAML."""
import argparse
import os
import sys
import yaml
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.training import train, train_time_marching


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # prefer explicit device if provided
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # pass the config path to training so it can be saved with checkpoints
    if cfg.get("training", {}).get("time_marching", False):
        train_time_marching(cfg, cfg_path=args.config)
    else:
        train(cfg, cfg_path=args.config)


if __name__ == "__main__":
    main()
