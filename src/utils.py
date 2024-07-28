import logging
import os
import re
from glob import glob
import matplotlib.pyplot as plt
import src.constants as c



def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def get_last_checkpoint_if_any(checkpoint_folder):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob('{}/*.h5'.format(checkpoint_folder), recursive=True)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]

def create_dir_and_delete_content(directory):
    os.makedirs(directory, exist_ok=True)
    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
        map(lambda f: os.path.join(directory, f), os.listdir(directory))),
        key=os.path.getmtime)
    # delete all but most current file to assure the latest model is availabel even if process is killed
    for file in files[:-4]:
        logging.info("removing old model: {}".format(file))
        os.remove(file)


def changefilename(path):
    files = os.listdir(path)
    for file in files:
        name=file.replace('-','_')
        lis = name.split('_')
        speaker = '_'.join(lis[:3])
        utt_id = '_'.join(lis[3:])
        newname = speaker + '-' +utt_id
        os.rename(path+'/'+file, path+'/'+newname)

def copy_wav(kaldi_dir,out_dir):
    import shutil
    from time import time
    orig_time = time()
    with open(kaldi_dir+'/utt2spk','r') as f:
        utt2spk = f.readlines()

    with open(kaldi_dir+'/wav.scp','r') as f:
        wav2path = f.readlines()

    utt2path = {}
    for wav in wav2path:
        utt = wav.split()[0]
        path = wav.split()[1]
        utt2path[utt] = path
    print(" begin to copy %d waves to %s" %(len(utt2path), out_dir))
    for i in range(len(utt2spk)):
        utt_id = utt2spk[i].split()[0].split('_')[:-1]  #utr2spk 中的 utt id 是'ZEBRA-KIDS0000000_1735129_26445a50743aa75d_00000 去掉后面的 _000
        utt_id = '_'.join(utt_id)
        speaker = utt2spk[i].split()[1]
        filepath = utt2path[utt_id]
                                                      #为了统一成和librispeech 格式一致 speaker与utt 用 '-'分割 speaker内部就用'_'
        target_filepath = out_dir + speaker.replace('-','_') + '-' + utt_id.replace('-','_') + '.wav'
        if os.path.exists(target_filepath):
            if i % 10 == 0: print(" No.:{0} Exist File:{1}".format(i, filepath))
            continue
        shutil.copyfile(filepath, target_filepath)

    print("cost time: {0:.3f}s ".format(time() - orig_time))


def plot_acc_eer_loss(train_file, val_file):
    epoch = []
    train_eer = []
    val_eer = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    with open(train_file) as f:
        lines = f.readlines()
        for line in lines:
            epoch.append(int(line.split(",")[0]))
            train_acc.append(float(line.split(",")[1]))
            train_eer.append(float(line.split(",")[2]))
            train_loss.append(float(line.split(",")[3]))

    with open(val_file) as f:
        lines = f.readlines()
        for line in lines:
            val_acc.append(float(line.split(",")[1]))
            val_eer.append(float(line.split(",")[2]))
            val_loss.append(float(line.split(",")[3]))

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epoch, train_eer, linestyle='--', marker='v', label='Training EER', color='red')
    plt.plot(epoch, val_eer, linestyle='--', marker='o', label='Validation EER', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    # plt.title('Accuracy and EER over Epochs')
    plt.title("A")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epoch, train_loss, linestyle='-', label='Training Loss', color='red')
    plt.plot(epoch, val_loss, linestyle=':', label='Validation Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.title('Loss over Epochs')
    plt.title("B")
    plt.legend()

    plt.tight_layout()
    plt.show()
