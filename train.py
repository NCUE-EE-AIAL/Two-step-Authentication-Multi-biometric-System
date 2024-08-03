
# train-clean-100: 251 speaker, 28539 utterance
# train-clean-360: 921 speaker, 104104 utterance
# test-clean: 40 speaker, 2620 utterance
# merged test: 80 speaker, 5323 utterance
# batchisize 32*3 : train on triplet: 5s - > 3.1s/steps , softmax pre train: 3.1 s/steps


import logging
from time import time
import numpy as np
import sys
import os
import random
from tensorflow.keras.optimizers import Adam
from keras.layers.core import Dense
from keras.models import Model

import src.constants as c
import src.select_batch as select_batch
from pre_process import data_catalog, preprocess_and_save
from src.models import rescnn_model
from src.random_batch import stochastic_mini_batch, MiniBatchGenerator
from src.triplet_loss import deep_speaker_loss
from src.utils import get_last_checkpoint_if_any, create_dir_and_delete_content, plot_acc_eer_loss
from test_model import eval_model


def create_dict(files,labels,spk_uniq):
    train_dict = {}
    for i in range(len(spk_uniq)):
        train_dict[spk_uniq[i]] = []

    for i in range(len(labels)):
        train_dict[labels[i]].append(files[i])

    for spk in spk_uniq:
        if len(train_dict[spk]) < 2:
            train_dict.pop(spk)
    unique_speakers=list(train_dict.keys())
    return train_dict, unique_speakers

def main(libri_dir=c.DATASET_DIR):

    logging.info('Looking for fbank features [.npy] files in {}.'.format(libri_dir))
    libri = data_catalog(libri_dir)

    if len(libri) == 0:
        logging.warning('Cannot find npy files, we will load audio, extract features and save it as npy file')
        logging.warning('please preprocess the data first through preprocess.py')

    # traing data select
    unique_speakers = libri['speaker_id'].unique()
    spk_utt_dict, unique_speakers = create_dict(libri['filename'].values, libri['speaker_id'].values, unique_speakers)
    select_batch.create_data_producer(unique_speakers, spk_utt_dict)

    #train data random
    generator = MiniBatchGenerator(libri, c.BATCH_SIZE, unique_speakers)

    # validation data random
    val_libri = data_catalog(c.TEST_DIR)
    val_unique_speakers = val_libri['speaker_id'].unique()
    val_generator = MiniBatchGenerator(val_libri, c.BATCH_SIZE, val_unique_speakers)

    batch = stochastic_mini_batch(libri, batch_size=c.BATCH_SIZE, unique_speakers=unique_speakers)
    batch_size = c.BATCH_SIZE * c.TRIPLET_PER_BATCH
    x, y = batch.to_inputs()
    b = x[0]
    num_frames = b.shape[0]
    train_batch_size = batch_size
    #batch_shape = [batch_size * num_frames] + list(b.shape[1:])  # A triplet has 3 parts.
    input_shape = (num_frames, b.shape[1], b.shape[2])
    logging.info('num_frames = {}'.format(num_frames))
    logging.info('batch size: {}'.format(batch_size))
    logging.info('input shape: {}'.format(input_shape))
    logging.info('x.shape : {}'.format(x.shape))
    orig_time = time()
    model = rescnn_model(input_shape=input_shape, batch_size=batch_size, num_frames=num_frames)
    logging.info(model.summary())
    grad_steps = 0
    current_epoch = 0

    last_checkpoint = get_last_checkpoint_if_any(c.CHECKPOINT_FOLDER)
    if last_checkpoint is not None:
        logging.info('Found checkpoint [{}]. Resume from here...'.format(last_checkpoint))
        model.load_weights(last_checkpoint)
        grad_steps = int(last_checkpoint.split('_')[-2])
        current_epoch = int(last_checkpoint.split('_')[-3])
        logging.info('[DONE]')

    # adam = Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer="adam", loss=deep_speaker_loss)
    print("model_build_time",time()-orig_time)
    logging.info('Starting training...')
    lasteer = 10
    eer = 1
    train_loss_list = []


    total_epoch = 60
    random_pretrain_epoch = 20
    step_per_epoch = len(libri) // batch_size
    while current_epoch <= total_epoch:
        logging.info('== Presenting step #{0}'.format(grad_steps))

        # random batch or select batch
        if current_epoch <= random_pretrain_epoch:
            x, y = next(iter(generator))
        else:
            x, _ = select_batch.best_batch(model, batch_size=c.BATCH_SIZE)
            y = np.random.uniform(size=(x.shape[0], 1))

        # If "ValueError: Error when checking target: expected ln to have shape (None, 512) but got array with shape (96, 1)"
        # please modify line to following line
        # y = np.random.uniform(size=(x.shape[0], 512))
        orig_time = time()

        train_loss_ = model.train_on_batch(x, y)
        train_loss_list.append(train_loss_)

        logging.info('== Processed in {0:.2f}s by the network, training loss = {1}.'.format(time() - orig_time, train_loss_))

        # VALIDATE EVERY EPOCH
        # to continue or measure result in every epoch
        if not grad_steps % step_per_epoch == 0:
            grad_steps += 1
            continue

        # calculate training results
        fm, acc, eer = [], [], []
        for i in range(3):
            fm_, _, acc_, eer_, _ = eval_model(model, train_batch_size, test_dir=c.DATASET_DIR, check_partial=True)
            fm.append(fm_)
            acc.append(acc_)
            eer.append(eer_)
        fm, acc, eer, train_loss = np.mean(fm), np.mean(acc), np.mean(eer), np.mean(train_loss_list)
        train_loss_list = []
        logging.info('training data EER = {0:.3f}, F-measure = {1:.3f}'.format(eer, fm))
        logging.info('Accuracy = {0:.3f}, Training Loss = {1:.3f}'.format(acc, train_loss))
        with open(c.CHECKPOINT_FOLDER + '/train_acc_eer_loss.txt', "a") as f:
            f.write("{0},{1},{2},{3}\n".format(current_epoch, acc, eer, train_loss))

        # calculate validation results
        val_loss_list = []
        for i in range(10):
            x, y = next(iter(val_generator))
            val_loss_ = model.evaluate(x, y, train_batch_size, verbose=0)
            val_loss_list.append(val_loss_)

        fm, recall, acc, eer, precision = [], [], [], [], []
        for i in range(3):
            fm_, recall_, acc_, eer_, precision_ = eval_model(model, train_batch_size, test_dir=c.TEST_DIR)
            fm.append(fm_)
            recall.append(recall_)
            acc.append(acc_)
            eer.append(eer_)
            precision.append(precision_)
        acc, eer = np.mean(acc), np.mean(eer)
        fm, recall, precision, val_loss = np.mean(fm), np.mean(recall), np.mean(precision), np.mean(val_loss_list)
        logging.info('== Testing model after batch #{0}'.format(grad_steps))
        logging.info('test EER = {0:.3f}, F-measure = {1:.3f}'.format(eer, fm))
        logging.info('Accuracy = {0:.3f}, Validation Loss = {1:.3f}'.format(acc, val_loss))
        with open(c.CHECKPOINT_FOLDER + '/val_acc_eer_loss.txt', "a") as f:
            f.write("{0},{1},{2},{3}\n".format(current_epoch, acc, eer, val_loss))
        with open(c.CHECKPOINT_FOLDER + '/val_prec_recall_fm.txt', "a") as f:
            f.write("{0},{1},{2},{3}\n".format(current_epoch, precision, recall, fm))

        # save checkpoint
        # checkpoints are really heavy so let's just keep the last one.
        create_dir_and_delete_content(c.CHECKPOINT_FOLDER)
        model.save_weights('{0}/model_{1}_{2}_{3:.5f}.h5'.format(c.CHECKPOINT_FOLDER, current_epoch, grad_steps, train_loss_))
        if eer < lasteer:
            files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith(".h5"),
                                  map(lambda f: os.path.join(c.BEST_CHECKPOINT_FOLDER, f), os.listdir(c.BEST_CHECKPOINT_FOLDER))),
                           key=lambda file: file.split('/')[-1].split('.')[-2], reverse=True)
            lasteer = eer
            for file in files[:-4]:
                logging.info("removing old model: {}".format(file))
                os.remove(file)
            model.save_weights(c.BEST_CHECKPOINT_FOLDER+'/best_model_{0}_{1}_{2:.5f}.h5'.format(current_epoch, grad_steps, eer))
        if current_epoch != random_pretrain_epoch and current_epoch <= random_pretrain_epoch:
            generator.on_epoch_end()
        elif current_epoch == random_pretrain_epoch + 1:
            generator.stop()

        grad_steps += 1
        current_epoch += 1



if __name__ == '__main__':
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')
    main()

    plot_acc_eer_loss("checkpoints/train_acc_eer_loss.txt", "checkpoints/val_acc_eer_loss.txt")
