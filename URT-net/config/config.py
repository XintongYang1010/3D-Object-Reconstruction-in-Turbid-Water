import argparse
import yaml
import os
import shutil

def get_parser():
    parser = argparse.ArgumentParser(description='IMAGE_DEMOIREING')
    parser.add_argument('--config', type=str, default='/home/hjc/Program/underwater/model/UHDM-main_L/config/uhdm_config_L_1.yaml', help='path to config file')
    args_cfg = parser.parse_args()
    assert args_cfg.config is not None
    with open(args_cfg.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for key in config:
        for k, v in config[key].items():
            setattr(args_cfg, k, v)

    return args_cfg


args = get_parser()


def save_config(args,target_path):

    source_file=args.config
    file_name = os.path.basename(source_file)
    # 定义目标文件路径
    target_file = os.path.join(target_path, file_name)

    # 复制文件
    shutil.copyfile(source_file, target_file)

    print(f'File copied from {source_file} to {target_file}')