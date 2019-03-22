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


def couple_indoors_images():
    clear_dir = "/home/csbhr/studysource/图像去雾/去雾数据集2/ITS_v2/clear"
    hazy_dir = "/home/csbhr/studysource/图像去雾/去雾数据集2/ITS_v2/hazy"

    clear_fnames = os.listdir(clear_dir)

    clear_full_path = []
    hazy_full_path = []

    for cf in clear_fnames:
        basename = cf.split(".")[0]
        match_hazy_full_path = glob.glob(os.path.join(hazy_dir, "{}*".format(basename)))
        fetch_nums = min(3, len(match_hazy_full_path))
        for i in range(fetch_nums):
            clear_full_path.append(os.path.join(clear_dir, cf))
            hazy_full_path.append(match_hazy_full_path[i])

    return clear_full_path, hazy_full_path


def indoors_action():
    train_dest_dir = "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train"
    val_dest_dir = "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/val"

    clear_full_path, hazy_full_path = couple_indoors_images()

    t_v_thre = int(len(clear_full_path) * 0.8)

    train_dict = {
        "clear_ori": clear_full_path[:t_v_thre],
        "hazy_ori": hazy_full_path[:t_v_thre]
    }
    train_dict["clear_dest"] = [
        os.path.join(train_dest_dir, "clear", "{0}.{1}".format(i, os.path.basename(co).split(".")[-1]))
        for i, co in enumerate(train_dict["clear_ori"])]
    train_dict["hazy_dest"] = [
        os.path.join(train_dest_dir, "hazy", "{0}.{1}".format(i, os.path.basename(ho).split(".")[-1]))
        for i, ho in enumerate(train_dict["hazy_ori"])]

    val_dict = {
        "clear_ori": clear_full_path[t_v_thre:],
        "hazy_ori": hazy_full_path[t_v_thre:]
    }
    train_num = len(train_dict["clear_ori"])
    val_dict["clear_dest"] = [
        os.path.join(val_dest_dir, "clear", "{0}.{1}".format(i + train_num, os.path.basename(co).split(".")[-1]))
        for i, co in enumerate(val_dict["clear_ori"])]
    val_dict["hazy_dest"] = [
        os.path.join(val_dest_dir, "hazy", "{0}.{1}".format(i + train_num, os.path.basename(ho).split(".")[-1]))
        for i, ho in enumerate(val_dict["hazy_ori"])]

    copy_files(train_dict["clear_ori"], train_dict["clear_dest"])
    copy_files(train_dict["hazy_ori"], train_dict["hazy_dest"])
    copy_files(val_dict["clear_ori"], val_dict["clear_dest"])
    copy_files(val_dict["hazy_ori"], val_dict["hazy_dest"])


indoors_action()

# def process_file_path(ori_path, train_dest_path, val_dest_path, prefix):
#     clear_filenames = os.listdir(os.path.join(ori_path, "clear"))
#
#     thre = int(len(clear_filenames) * 0.8)
#     train_clear_filenames = clear_filenames[:thre]
#     val_clear_filenames = clear_filenames[thre:]
#
#     train_hazy_filenames = []
#     val_hazy_filenames = []
#
#     for t_fname in train_clear_filenames:
#         basename = t_fname.split(".")[0]
#         train_hazy_path = glob.glob(os.path.join(os.path.join(ori_path, "hazy"), basename + "*"))
#         train_hazy_filenames.extend([os.path.basename(p) for p in train_hazy_path])
#     for v_fname in val_clear_filenames:
#         basename = v_fname.split(".")[0]
#         val_hazy_path = glob.glob(os.path.join(os.path.join(ori_path, "hazy"), basename + "*"))
#         val_hazy_filenames.extend([os.path.basename(p) for p in val_hazy_path])
#
#     full_paths = {}
#
#     full_paths["train_clear_ori"] = [os.path.join(ori_path, "clear", fn) for fn in train_clear_filenames]
#     full_paths["train_clear_dest"] = [os.path.join(train_dest_path, "clear", prefix + "_" + fn) for fn in
#                                       train_clear_filenames]
#     full_paths["train_hazy_ori"] = [os.path.join(ori_path, "hazy", fn) for fn in train_hazy_filenames]
#     full_paths["train_hazy_dest"] = [os.path.join(train_dest_path, "hazy", prefix + "_" + fn) for fn in
#                                      train_hazy_filenames]
#     full_paths["val_clear_ori"] = [os.path.join(ori_path, "clear", fn) for fn in val_clear_filenames]
#     full_paths["val_clear_dest"] = [os.path.join(val_dest_path, "clear", prefix + "_" + fn) for fn in
#                                     val_clear_filenames]
#     full_paths["val_hazy_ori"] = [os.path.join(ori_path, "hazy", fn) for fn in val_hazy_filenames]
#     full_paths["val_hazy_dest"] = [os.path.join(val_dest_path, "hazy", prefix + "_" + fn) for fn in val_hazy_filenames]
#
#     return full_paths

# outdoors_ori_path = "/home/csbhr/studysource/图像去雾/去雾数据集2/SOTS/outdoor"
# indoors_ori_path = "/home/csbhr/studysource/图像去雾/去雾数据集2/ITS_v2"
# train_dest_path = "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/train"
# val_dest_path = "/home/csbhr/workspace/python/python_data/DeHaze_CGAN_ResNet/val"
#
# # outdoor
# outdoors_full_paths = process_file_path(outdoors_ori_path, train_dest_path, val_dest_path, "1")
# copy_files(outdoors_full_paths["train_clear_ori"], outdoors_full_paths["train_clear_dest"])
# copy_files(outdoors_full_paths["train_hazy_ori"], outdoors_full_paths["train_hazy_dest"])
# copy_files(outdoors_full_paths["val_clear_ori"], outdoors_full_paths["val_clear_dest"])
# copy_files(outdoors_full_paths["val_hazy_ori"], outdoors_full_paths["val_hazy_dest"])
#
# # indoor
# indoors_full_paths = process_file_path(indoors_ori_path, train_dest_path, val_dest_path, "2")
# copy_files(indoors_full_paths["train_clear_ori"], indoors_full_paths["train_clear_dest"])
# copy_files(indoors_full_paths["train_hazy_ori"], indoors_full_paths["train_hazy_dest"])
# copy_files(indoors_full_paths["val_clear_ori"], indoors_full_paths["val_clear_dest"])
# copy_files(indoors_full_paths["val_hazy_ori"], indoors_full_paths["val_hazy_dest"])
