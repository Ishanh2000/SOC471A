
# CONSTANTS YOU NEED TO MODIFY

#whether to train on GPU
CUDA=True 
# root directory that contains the training/testing data
DATA_HOME = "/home/ishanhmisra/AUM/SOC471A_Project/conflict_data/prediction" #"/dfs/scratch0/reddit/conflict/prediction"
LOG_DIR = "/home/ishanhmisra/AUM/SOC471A_Project/conflict_data/prediction" #"/dfs/scratch0/reddit/conflict/prediction"
#whether to show results on the test set
PRINT_TEST=False

# CONSTANTS YOU MAY WANT TO MODIFY (BUT DON"T NEED TO)
TRAIN_DATA=DATA_HOME+"/preprocessed_train_data.pkl"
VAL_DATA=DATA_HOME+"/preprocessed_val_data.pkl"
TEST_DATA=DATA_HOME+"/preprocessed_test_data.pkl"
BATCH_SIZE=512
#NOTE: THESE PREPROCESSED FILES HAVE A FIXED BATCH SIZE

WORD_EMBEDS=DATA_HOME+"/embeddings/glove_word_embeds.txt"

USER_EMBEDS=DATA_HOME+"/embeddings/user_vecs.npy"
USER_IDS=DATA_HOME+"/embeddings/user_vecs.vocab"

SUBREDDIT_EMBEDS=DATA_HOME+"/embeddings/sub_vecs.npy"
SUBREDDIT_IDS=DATA_HOME+"/embeddings/sub_vecs.vocab"

POST_INFO=DATA_HOME+"/detailed_data/post_crosslink_info.tsv"
LABEL_INFO=DATA_HOME+"/detailed_data/label_info.tsv"
PREPROCESSED_DATA=DATA_HOME+"/detailed_data/tokenized_posts.tsv"
ANALYSIS_SECONDARY_DATA = "./data_secondary/"
ANALYSIS_SECONDARY_IMGS = "./img/"

VOCAB_SIZE = 174558
NUM_USERS = 118381
NUM_SUBREDDITS = 51278
WORD_EMBED_DIM = 300
METAFEAT_LEN = 263
NUM_CLASSES = 1
MAX_LEN=50

SUMMARY_REQ = {
  "REL_RATE_POS" : 1.6,
  "REL_RATE_NEG" : 8.8,
  "NUM_MOBILIZ" : 22075,
  "CORR_CONFLICT_SIMILAR" : 0.51,
  "CORR_CONFLICT_DIFFERENT" : 0.34,
  "CORR_ANGRY_WORDS_ATT" : 0.31,
  "CORR_ANGRY_WORDS_DEF" : 0.26,
  "REL_ANGRY_WORDS" : 1.44,
  "REL_DELETION_NEG" : 0.205,
  "REL_DELETION_POS" : 0.008,
  "REL_FREQ_ATT" : 2.0,
  "REL_FREQ_DEF" : 20.0,
  "CORR_ANGRY_WORDS_ATT_DEF" : 0.015,
  "CORR_ANGRY_WORDS_ATT_ATT" : 0.011,
  "CORR_ANGRY_WORDS_DEF_ATT" : 0.017,
  "CORR_ANGRY_WORDS_DEF_DEF" : 0.014,
  "CORR_DEF_REPLY_ATT_SUCCESS" : 0.97,
  "CORR_A_PGRNK_DEF_SUCCESS" : 0.036,
  "CORR_A_PGRNK_DEF_UNSUCCESS" : 0.031,
  "CORR_D_PGRNK_DEF_SUCCESS" : 0.052,
  "CORR_D_PGRNK_DEF_UNSUCCESS" : 0.028,
}

