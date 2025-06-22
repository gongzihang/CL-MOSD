import os
import json
import yaml
from datetime import datetime
from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf, DictConfig
import argparse

def convert_to_dict(obj):
    """
    支持多种类型：Namespace, OmegaConf, dict 等转为普通 dict
    """
    if isinstance(obj, dict):
        return obj
    elif isinstance(obj, SimpleNamespace) or isinstance(obj, argparse.Namespace):
        return vars(obj)
    elif isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    else:
        raise TypeError(f"Unsupported config type: {type(obj)}")

def save_full_config(cfg, args, save_dir="./DEMO", filename="config.json", writer: SummaryWriter = None):
    os.makedirs(save_dir, exist_ok=True)

    # 转成 dict
    cfg_dict = convert_to_dict(cfg) if cfg is not None else {}
    args_dict = convert_to_dict(args) if args is not None else {}

    # 合并并添加元信息
    full_config = {
        "cfg": cfg_dict,
        "args": args_dict,
        "_meta": {
            "saved_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # 保存为 JSON 文件
    json_path = os.path.join(save_dir, filename)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=4, ensure_ascii=False)

    # 写入 TensorBoard（可选）
    if writer is not None:
        config_yaml = yaml.dump(full_config, allow_unicode=True)
        writer.add_text("full_config", f"```yaml\n{config_yaml}\n```")

    print(f"配置已保存至: {json_path}")
    if writer is not None:
        print("配置也写入到了 TensorBoard（Text 标签页）")

