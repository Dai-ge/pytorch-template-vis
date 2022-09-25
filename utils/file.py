import time
import logging
import os
import yaml

def mkdir(path):
    """make a new dirs if the dirs do not exists
        else do nothing
    Args:
        path (str): the path user want to make dirs if not exists
    """
    if not os.path.exists(path):
        os.makedirs(path)

def yaml_loader(yaml_path):
    content = None
    with open(yaml_path,'r') as f:
        content = yaml.safe_load(f)
    return content

def setting_loader(yaml_path):
    all_settings = None
    model_setting = None
    train_setting = None
    data_setting = None

    with open(yaml_path,'r') as f:
        all_settings = yaml.safe_load(f)
    
    model_setting = all_settings['model_setting']
    train_setting = all_settings['train_setting']
    data_setting = all_settings['data_setting']

    return all_settings,model_setting,train_setting,data_setting



if __name__ == "__main__":
    print(yaml_loader(r'D:\VSCodeLib\Pytorch_Template\tasks\configs\STCoodNet.yml'))
    pass
