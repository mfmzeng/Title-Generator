"""
BERT Encoder–Decoder chapter title generator (seq2seq).

Fine-tunes Hugging Face EncoderDecoderModel (BERT encoder + BertGeneration decoder).
"""

# Model — bert-base + fp16 fits most 4 GB GPUs; lower MAX_INPUT_LENGTH if OOM
ENCODER_MODEL = "google-bert/bert-base-uncased"
DECODER_MODEL = "google-bert/bert-base-uncased"
CHECKPOINT_DIR = "checkpoints/bert2bert-titles"

# Instruction-tuning prefix (encoder input)
INPUT_PREFIX = "generate chapter title: "

# Sequence lengths
MAX_INPUT_LENGTH = 384
MAX_TARGET_LENGTH = 48

# Training
# Use a short first pass for smoke testing; raise these later for full training.
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
FP16 = True
GRADIENT_CHECKPOINTING = True

# Dataset — CMU Book Summaries (summary text → book title)
DATASET_ID = "textminr/cmu-book-summaries"
VAL_FRACTION = 0.1
RANDOM_SEED = 42
MAX_ROWS = None  # None = use full dataset (~16.6k rows)
MAX_TRAIN_SAMPLES = 2000
MAX_VAL_SAMPLES = 200
CHUNK_CHARS = 3_000

# Generation
NUM_BEAMS = 4
NUM_CANDIDATES = 3
MAX_GEN_LENGTH = 48

# Safety
ENABLE_PROFANITY_FILTER = True
BLOCKED_WORDS = (
    "damn",
    "hell",
    "shit",
    "fuck",
    "bitch",
    "asshole",
    "bastard",
)
