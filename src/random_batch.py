import numpy as np
import pandas as pd
import os
import json
import numpy as np
import pandas as pd
import threading
from queue import Queue
from time import sleep, time

import src.constants as c
from pre_process import data_catalog


def clipped_audio(x, num_frames=c.NUM_FRAMES):
    if x.shape[0] > num_frames:
        bias = np.random.randint(0, x.shape[0] - num_frames)
        clipped_x = x[bias: num_frames + bias]
    else:
        clipped_x = x

    return clipped_x


class MiniBatchGenerator:
    def __init__(self, libri, unique_speakers=None, batch_size=c.BATCH_SIZE, num_frames=c.NUM_FRAMES, queue_size=c.RANDOM_QUEUE_SIZE, num_producers=c.NUM_PRODUCERS):
        self.libri = libri
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.unique_speakers = unique_speakers if unique_speakers is not None else list(libri['speaker_id'].unique())
        self.indices = np.arange(len(libri))
        self.queue = Queue(maxsize=queue_size)
        self.num_producers = num_producers
        self.stop_event = threading.Event()
        self.create_data_producer()
        self.on_epoch_end()


    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def create_data_producer(self):
        for _ in range(self.num_producers):
            thread = threading.Thread(target=self.add_to_queue)
            thread.setDaemon(True)
            thread.start()

    def add_to_queue(self):
        while not self.stop_event.is_set():
            if self.queue.full():
                sleep(0.1)
                continue

            # create batch
            batch_indices = np.random.choice(self.indices, self.batch_size, replace=False)
            batch = self._create_batch(batch_indices)
            self.queue.put(batch)

    def get_batch(self):
        return self.queue.get()

    def _create_batch(self, batch_indices):
        anchor_batch = None
        positive_batch = None
        negative_batch = None

        for idx in batch_indices:
            anchor_positive_speaker = self.libri.iloc[idx]['speaker_id']
            negative_speaker = np.random.choice([sp for sp in self.unique_speakers if sp != anchor_positive_speaker])

            anchor_positive_file = self.libri[self.libri['speaker_id'] == anchor_positive_speaker].sample(n=2, replace=False)
            negative_file = self.libri[self.libri['speaker_id'] == negative_speaker].sample(n=1)

            anchor_df = pd.DataFrame(anchor_positive_file[0:1])
            anchor_df['training_type'] = 'anchor'
            positive_df = pd.DataFrame(anchor_positive_file[1:2])
            positive_df['training_type'] = 'positive'
            negative_df = pd.DataFrame(negative_file)
            negative_df['training_type'] = 'negative'

            if anchor_batch is None:
                anchor_batch = anchor_df.copy()
            else:
                anchor_batch = pd.concat([anchor_batch, anchor_df], axis=0)
            if positive_batch is None:
                positive_batch = positive_df.copy()
            else:
                positive_batch = pd.concat([positive_batch, positive_df], axis=0)
            if negative_batch is None:
                negative_batch = negative_df.copy()
            else:
                negative_batch = pd.concat([negative_batch, negative_df], axis=0)

        libri_batch = pd.concat([anchor_batch, positive_batch, negative_batch], axis=0)

        new_x = []
        for i in range(len(libri_batch)):
            filename = libri_batch[i:i + 1]['filename'].values[0]
            x = np.load(filename)
            new_x.append(clipped_audio(x))
        x = np.array(new_x)
        y = libri_batch['speaker_id'].values

        return x, y

    def stop(self):
        self.stop_event.set()
        while not self.queue.empty():
            self.queue.get()
