
DATASET_DIR = "audio/LibriSpeech/train-clean-360-npy/"
TEST_DIR = 'audio/LibriSpeech/test-clean-npy'
WAV_DIR = "audio/LibriSpeech/train-clean-100/"
KALDI_DIR = ''
ADD_NOISE = False

BATCH_SIZE = 32        #must be even
TRIPLET_PER_BATCH = 3
CANDIDATES_PER_BATCH = 640       # 18s per batch

TEST_NEGATIVE_No = 99

NUM_FRAMES = 160   # 299 - 16*2
SAMPLE_RATE = 16000
TRUNCATE_SOUND_SECONDS = (0.2, 1.81)  # (start_sec, end_sec)
ALPHA = 0.1
HIST_TABLE_SIZE = 10
NUM_SPEAKERS = 251

# select and random batch
# CPU: 4 cores (related to NUM_PRODUCERS), RAM: 32GB (related to QUEUE_SIZE)
SELECT_QUEUE_SIZE = 12
RANDOM_QUEUE_SIZE = 12
NUM_PRODUCERS = 4

CHECKPOINT_FOLDER = 'checkpoints/'
BEST_CHECKPOINT_FOLDER = 'checkpoints/best_checkpoint/'
PRE_CHECKPOINT_FOLDER = 'checkpoints/pretraining_checkpoints/'
