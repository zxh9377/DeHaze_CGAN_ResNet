import os
import shutil
import glob


def copy_files(origin_file_paths, dest_file_paths):
    '''
    copy files from origin path to dest path
    :param origin_file_paths: a list, origin files' full path
    :param dest_file_paths: a list, dest files' full path
    :return:
    '''
    for ori, dest in zip(origin_file_paths, dest_file_paths):
        shutil.copyfile(ori, dest)
        print("-- copy file {} ...".format(os.path.basename(ori)))


def process_file_path(ori_path, train_dest_path, val_dest_path, prefix):
    clear_filenames = os.listdir(os.path.join(ori_path, "clear"))

    thre = int(len(clear_filenames) * 0.8)
    train_clear_filenames = clear_filenames[:thre]
    val_clear_filenames = clear_filenames[thre:]

    train_hazy_filenames = []
    val_hazy_filenames = []

    for t_fname in train_clear_filenames:
        basename = t_fname.split(".")[0]
        train_hazy_path = glob.glob(os.path.join(os.path.join(ori_path, "hazy"), basename + "*"))
        train_hazy_filenames.extend([os.path.basename(p) for p in train_hazy_path])
    for v_fname in val_clear_filenames:
        basename = v_fname.split(".")[0]
        val_hazy_path = glob.glob(os.path.join(os.path.join(ori_path, "hazy"), basename + "*"))
        val_hazy_filenames.extend([os.path.basename(p) for p in val_hazy_path])

    full_paths = {}

    full_paths["train_clear_ori"] = [os.path.join(ori_path, "clear", fn) for fn in train_clear_filenames]
    full_paths["train_clear_dest"] = [os.path.join(train_dest_path, "clear", prefix + "_" + fn) for fn in
                                      train_clear_filenames]
    full_paths["train_hazy_ori"] = [os.path.join(ori_path, "hazy", fn) for fn in train_hazy_filenames]
    full_paths["train_hazy_dest"] = [os.path.join(train_dest_path, "hazy", prefix + "_" + fn) for fn in
                                     train_hazy_filenames]
    full_paths["val_clear_ori"] = [os.path.join(ori_path, "clear", fn) for fn in val_clear_filenames]
    full_paths["val_clear_dest"] = [os.path.join(val_dest_path, "clear", prefix + "_" + fn) for fn in
                                    val_clear_filenames]
    full_paths["val_hazy_ori"] = [os.path.join(ori_path, "hazy", fn) for fn in val_hazy_filenames]
    full_paths["val_hazy_dest"] = [os.path.join(val_dest_path, "hazy", prefix + "_" + fn) for fn in val_hazy_filenames]

    return full_paths


outdoors_ori_path = "/home/csbhr/studysource/图像去雾/去雾数据集2/SOTS/outdoor"
indoors_ori_path = "/home/csbhr/studysource/图像去雾/去雾数据集2/ITS_v2"
train_dest_path = "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train"
val_dest_path = "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/val"

# outdoor
# outdoors_full_paths = process_file_path(outdoors_ori_path, train_dest_path, val_dest_path, "1")
# copy_files(outdoors_full_paths["train_clear_ori"], outdoors_full_paths["train_clear_dest"])
# copy_files(outdoors_full_paths["train_hazy_ori"], outdoors_full_paths["train_hazy_dest"])
# copy_files(outdoors_full_paths["val_clear_ori"], outdoors_full_paths["val_clear_dest"])
# copy_files(outdoors_full_paths["val_hazy_ori"], outdoors_full_paths["val_hazy_dest"])

# indoor
indoors_full_paths = process_file_path(indoors_ori_path, train_dest_path, val_dest_path, "2")
copy_files(indoors_full_paths["train_clear_ori"], indoors_full_paths["train_clear_dest"])
copy_files(indoors_full_paths["train_hazy_ori"], indoors_full_paths["train_hazy_dest"])
copy_files(indoors_full_paths["val_clear_ori"], indoors_full_paths["val_clear_dest"])
copy_files(indoors_full_paths["val_hazy_ori"], indoors_full_paths["val_hazy_dest"])
