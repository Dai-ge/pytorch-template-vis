import time
import logging
import shutil
import os
import yaml


#---------------------Basic Methods---------------------#
#These methods are fundamental part for advance methods #
#Maintain these methods carefully                       #
#-------------------------------------------------------#

def ensure_dir(dir_path,verbose = False):
    """Ensure the dir exist, if not exist, create it in dir_path

    Args:
        dir_path (str): the dir folder path you want to create
        verbose (bool, optional): Show Info about creating or not. Defaults to False.
    """
    if not os.path.exists(dir_path):
        if verbose: print(f'{dir_path} not exists, create the dir')
        os.mkdir(dir_path)
    else:
        if verbose: print(f'{dir_path} exists, no need to create the dir')

def del_filesInDir(dir_path):
    """Delete the all the files in dir_path

    Args:
        dir_path (str): the dir folder which you want to delete all the files in that folder
    """
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path,f))

def yaml_loader(yaml_path):
    """Load yaml config file

    Args:
        yaml_path (str): the yaml file path

    Returns:
        content: the yaml setting
    """
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

#--------------------Advance Methods--------------------#
#  These methods are based on Basic Methods             #
#  Maintain these methods together with Basic Methods   #
#-------------------------------------------------------#

def TypeA2TypeB(dir_path,saved_dir_path,typeA='.dat',typeB='.raw'):
    ensure_dir(saved_dir_path) #create the saved dir path if the path do not exists
    del_filesInDir(saved_dir_path)
   
    for root,dirs,files in os.walk(dir_path):
        for file_name in files:
            if(os.path.splitext(file_name)[-1] == typeA): #to avoid other types of files
                dat_file_path = os.path.join(root,file_name)
                raw_file_path = os.path.join(saved_dir_path,file_name)
                shutil.copyfile(dat_file_path,raw_file_path) #first copy the dat file to RAW dir

    for root,dirs,files in os.walk(saved_dir_path):
        for file_name in files:
            if(os.path.splitext(file_name)[-1] == typeA): #just to avoid conner case
                dat_file = os.path.join(root,file_name)
                raw_file = os.path.join(root,os.path.splitext(file_name)[0]+typeB)
                os.rename(dat_file,raw_file)







if __name__ == "__main__":
    print(yaml_loader(r'D:\VSCodeLib\Pytorch_Template\tasks\configs\STCoodNet.yml'))
    pass
