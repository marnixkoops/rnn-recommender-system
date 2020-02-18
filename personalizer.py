"""
# [+] PROJECT INFO
#     - Recurrent Neural Network model based on implicit feedback data (product views)
#     - Production ready framework with run options, logging and experiment mode
#     - Main purpose is to predict multiple interesting items on customer level (cookie_id)
#
# Owner: Marnix Koops / marnix.koops@coolblue.nl / marnixkoops@gmail.com
"""

# ==================================================================================================
# [+] MODULE SETUP
# ==================================================================================================


import os
import glob
import re
import argparse
import tempfile
import shutil
import time
import datetime
import warnings
import gc
import logging
import mlflow
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from google.cloud import bigquery, bigquery_storage

warnings.simplefilter(action="ignore", category=UserWarning)  # no Google Cloud credential warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# ==================================================================================================
# [+] RUN SETTINGS & CONSTANTS
# ==================================================================================================

# cli argument parsing
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--mode",
#     default="production",
#     help="'production' trains and predicts on full data, 'experiment' evaluates on a hold out set",
#     type=str,
# )
# parser.add_argument(
#     "--train",
#     default=True,
#     help="if True will train model and predict. Else will try to download latest model from bucket and predict",
#     type=bool,
# )
# parser.add_argument(
#     "--evaluate", default=True, help="if True will compute prediction metrics", type=bool
# )
# parser.add_argument(
#     "--log", default=True, help="if True will log mlflow results to the bucket", type=bool
# )
# parser.add_argument(
#     "--dry_run", default=False, help="if True will run process on subset of data", type=bool
# )
# args = parser.parse_args()
#
# # run settings
# RUN_MODE = args.mode  # one of ['production', 'experiment']
# TRAIN_MODEL = args.train  # if False, will load most recent model object instead of training
# EVALUATE_MODEL = args.evaluate  # if False, will create predictions without further evaluation
# DRY_RUN = args.dry_run  # runs flow on small subset of data for speed and disables mlfow tracking
# LOGGING = args.log  # mlflow experiment logging

PREDICTION_LEVEL = "product"
RUN_MODE = "experiment"  # one of ['production', 'experiment']
TRAIN_MODEL = True  # if False, will load most recent model object instead of retraining
EVALUATE_MODEL = True  # if False, will create predictions without further evaluation
DRY_RUN = False  # runs flow on small subset of data for speed and disables mlfow tracking
LOGGING = True  # mlflow experiment logging

SAVE_MODEL_TO_DISK = True  # if true saves trained model to disk
SAVE_MODEL_TO_GCS = True  # if true saves trained model to bucket
SAVE_LOGS_TO_GCS = True  # if true uploads log file to bucket
SAVE_RESULTS_TO_BQ = False  # if true appends results to BigQuery table

DATETIME = datetime.datetime.now().replace(microsecond=0)  # timestamp of run
DATETIME = re.sub(r"([\s+:])", r"-", "{}".format(DATETIME))  # replace spaces and : with -
SEED = 808  # seed for reproducibility


# directory related
PROJECT = "coolblue-bi-data-science-exp"
DATASET = "producttype_personalization"
BUCKET = "neural-product-personalization"
MODEL_DIR = "./neural-product-personalization/model"
LOG_DIR = "./neural-product-personalization/log"
for directory in [MODEL_DIR, LOG_DIR]:  # make local directories if they do not exist
    if not os.path.exists(directory):
        os.makedirs(directory)

# data related
N_WEEKS_OF_DATA = 2
N_ITEMS = 12000
MIN_ITEMS_TRAIN = 2  # sequences with less products (excluding target) are invalid and removed
MIN_ITEMS_PRED = 0  # 0 + 1 (target) == require a single product view to predict on
N_DAYS_TO_PREDICT = 1  # how many days to holdout to create predictions on in experiments
FILTER_REPEATED_UNIQUES = True  # filters sequences with only 1 unique item e.g. [1, 1, 1, 1]
WINDOW_LEN = 4  # fixed moving window size for generating input-sequence/target rows for training
PRED_LOOKBACK = 8  # number of most recent products used per sequence in the test set to predict on
N_ITEMS_TO_PRED = 10  # number of top N item probabilities to extract from the full item matrix

# model related
EMBED_DIM = 48  # number of dimensions for the embeddings
N_HIDDEN_UNITS = 84  # number of units in the GRU layers
MAX_EPOCHS = 48  # maximum number of epochs to train for
BATCH_SIZE = 512  # batch size for training
DROPOUT = 0.0  # network node dropout (better to avoid in RNNs)
RECURRENT_DROPOUT = 0.10  # recurrent state dropout (during training only)
LEARNING_RATE = 0.005
OPTIMIZER = tf.keras.optimizers.Adamax(learning_rate=LEARNING_RATE)
REGULARIZER = None  # tf.keras.regularizers.l2(0.01)

# FP16 mixed-precision training instead of FP32 for gradients and model weights (only on GPU)
# See: https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#tensorflow-amp
# Needs more investigation in terms of speed, gives a warning for memory heavy tensor conversion
# OPTIMIZER = tf.train.experimental.enable_mixed_precision_graph_rewrite(OPTIMIZER)

# training related
VAL_RATIO = 0.25  # percentage of data to use for validation during model training
SHUFFLE_TRAIN = True  # shuffles only the train set
SHUFFLE_TRAIN_AND_VAL = True  # shuffles both the train and validation set
DATA_IMBALANCE_CORRECTION = False  # supply items weights to correct imbalance during training

# dry run related
if DRY_RUN:
    TRAIN_MODEL = True
    MAX_EPOCHS = 1
    BATCH_SIZE = 32
    MIN_ITEMS_TRAIN = 1
    FILTER_REPEATED_UNIQUES = False

# logging related
logging.basicConfig(  # logging to terminal & disk file
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] [%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler(
            "{}/{}_{}.log".format("./neural-product-personalization/log", "logfile", DATETIME)
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger()

logger.info(
    "[+] Starting process on {} in project {} and bucket {}".format(DATETIME, PROJECT, BUCKET)
)
logger.info(
    "     Run settings are RUN_MODE {} DRY_RUN {}, TRAIN_MODEL {}, EVALUATE_MODEL {}, LOGGING {}, WORK_DIR {}".format(
        RUN_MODE, DRY_RUN, TRAIN_MODEL, EVALUATE_MODEL, LOGGING, os.getcwd()
    )
)

# ==================================================================================================
# [+] TENSORFLOW SETTINGS
# ==================================================================================================

tf.keras.backend.clear_session()  # clear potentially remaining network graphs in the memory
tf.random.set_seed(SEED)  # set seed for reproducibility
# is_gpu_available = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# tf.config.list_physical_devices("GPU")

# define on which device we are running TensorFlow (GPU/CPU)
# if on GPU, force memory growth options and increase limit to avoid memory allocation issues
all_devices = str(device_lib.list_local_devices())
gpu_devices = tf.config.experimental.list_physical_devices("XLA_GPU")
if "Tesla P100" in all_devices:
    DEVICE = "Tesla P100 GPU"
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)  # no allocating memory upfront
    tf.config.experimental.set_virtual_device_configuration(
        gpu_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)]
    )
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
elif "GPU" in all_devices:
    DEVICE = "GPU"
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    tf.config.experimental.set_virtual_device_configuration(
        gpu_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)]
    )
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
elif "CPU" in all_devices:
    DEVICE = "CPU"

logger.info("[+] Running TensorFlow version {} on {}".format(tf.__version__, DEVICE))

# ==================================================================================================
# [+] INPUT DATA QUERIES
# ==================================================================================================


def query_product_sequence_data():
    """
    Queries complete sessions
    """
    logger.info("     Downloading sequence data for training")
    query = """
    SELECT
      coolblue_cookie_id,
      visit_date,
      product_sequence
    FROM
      `{}.{}.daily_sequences`
    """.format(
        PROJECT, DATASET
    )
    client = bigquery.Client(project=PROJECT)
    bqstorage_client = bigquery_storage.BigQueryStorageClient()
    query_result = client.query(query).result()
    df = query_result.to_dataframe(bqstorage_client=bqstorage_client, progress_bar_type="tqdm")
    return df


def query_product_sequence_with_intraday_data():
    """
    Queries complete sessions and adds intraday sessions and joins to one table
    """
    logger.info("     Downloading sequence data including intradays for training")
    query = """
    SELECT
      coolblue_cookie_id_intra AS coolblue_cookie_id,
      visit_date_intra AS visit_date,
      COALESCE(CONCAT(product_sequence, ',', product_sequence_intra), product_sequence_intra, product_sequence) AS product_sequence,
    FROM
      `{}.{}.daily_sequences` AS sequences
    RIGHT JOIN
      `{}.{}.intradaily_sequences` AS sequences_intraday
    ON
      sequences.coolblue_cookie_id = sequences_intraday.coolblue_cookie_id_intra
    """.format(
        PROJECT, DATASET, PROJECT, DATASET
    )
    client = bigquery.Client(project=PROJECT)
    bqstorage_client = bigquery_storage.BigQueryStorageClient()
    query_result = client.query(query).result()
    df = query_result.to_dataframe(bqstorage_client=bqstorage_client, progress_bar_type="tqdm")
    return df


def query_product_map_data():
    """
    Queries the data to map product_ids to product_type_ids
    """
    logger.info("     Downloading product mapping data from BigQuery")
    query = """
    SELECT
      product_id,
      product_name,
      product_type_id,
      product_type_name
    FROM
      `coolblue-bi-platform-prod.product.products`
    """
    client = bigquery.Client(project=PROJECT)
    bqstorage_client = bigquery_storage.BigQueryStorageClient()
    query_result = client.query(query).result()
    df = query_result.to_dataframe(bqstorage_client=bqstorage_client, progress_bar_type="tqdm")
    return df


# ==================================================================================================
# [+] FUNCTION DEFINITIONS
# ==================================================================================================


def upload_file_to_gcs(project, bucket, file_location, destination_file_location):
    if type(bucket) is not Bucket:
        client = storage.Client(project=project)
        bucket = client.get_bucket(bucket)

    blob = bucket.blob(destination_file_location)
    blob.chunk_size = 1 << 29  # increase chunk size for faster uploading
    blob.upload_from_filename(file_location)
    return True


def download_file_from_gcs(project, bucket_name, file_location, destination_file_location=None):
    gs_location = "gs://{}/{}".format(bucket_name, file_location)

    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket_name)

    blob = bucket.get_blob(file_location)
    if not blob:
        raise FileNotFoundError("{} does not exist!".format(gs_location))

    if not destination_file_location:
        destination_file_location = tempfile.NamedTemporaryFile(delete=False).name

    blob.chunk_size = 1 << 29  # increase chunk size for faster downloading
    blob.download_to_filename(destination_file_location)
    return destination_file_location


def save_trained_model_to_disk():
    model.save("./neural-product-personalization/model/model.h5")
    with open("./neural-product-personalization/model/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def upload_trained_model_to_gcs(project=PROJECT, bucket=BUCKET):
    save_trained_model_to_disk()  # first save model to disk
    model_file_size = os.path.getsize("./neural-product-personalization/model/model.h5")
    logger.info(
        "   Writing trained model to disk, filesize: {} KB".format(
            round(model_file_size / (1024.0), 3)
        )
    )
    logger.info("   Uploading trained model to GCS: {}/{}/model/".format(PROJECT, BUCKET))
    upload_file_to_gcs(
        project, bucket, "./neural-product-personalization/model/model.h5", "model/model.h5"
    )


def download_trained_model_from_gcs(project=PROJECT, bucket=BUCKET):
    model_file_location = download_file_from_gcs(
        project, bucket, "model/model.h5", "./neural-product-personalization/model/model.h5"
    )
    model = tf.keras.models.load_model(model_file_location)
    return model


def upload_tokenizer_to_gcs(project=PROJECT, bucket=BUCKET):
    logger.info("   Uploading tokenizers to GCS: {}/{}/model/".format(PROJECT, BUCKET))
    with open("./neural-product-personalization/model/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    upload_file_to_gcs(
        project,
        bucket,
        "./neural-product-personalization/model/tokenizer.pickle",
        "model/tokenizer.pickle",
    )

    with open("./neural-product-personalization/model/tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    upload_file_to_gcs(
        PROJECT,
        BUCKET,
        "./neural-product-personalization/model/tokenizer.pickle",
        "model/tokenizer.pickle",
    )


def download_tokenizer_from_gcs(project=PROJECT, bucket=BUCKET):
    tokenizer_step1_file_location = download_file_from_gcs(
        project,
        bucket,
        "model/tokenizer.pickle",
        "./neural-product-personalization/model/tokenizer.pickle",
    )
    with open(tokenizer_step1_file_location, "rb") as handle:
        tokenizer = pickle.load(handle)

    return tokenizer


def upload_log_to_gcs(project=PROJECT, bucket=BUCKET):
    logger.info("   Uploading log to GCS: {}/{}/model/".format(PROJECT, BUCKET))
    upload_file_to_gcs(
        project,
        bucket,
        "./neural-product-personalization/log/logfile_{}.log".format(DATETIME),
        "log/logfile_{}.log".format(DATETIME),
    )


def zip_and_upload_mlruns_to_gcs(project=PROJECT, bucket=BUCKET):
    shutil.make_archive(
        "./neural-product-personalization/mlruns_{}".format(DATETIME),
        "zip",
        "./neural-product-personalization/mlruns",
    )  # create local zip archive
    logger.info("   Uploading zipped mlruns folder to GCS: {}/{}/mlruns/".format(PROJECT, BUCKET))
    upload_file_to_gcs(
        project,
        bucket,
        "./neural-product-personalization/mlruns_{}.zip".format(DATETIME),
        "mlruns/mlruns_{}.zip".format(DATETIME),
    )


def upload_results_to_bigquery(
    df: pd.DataFrame, project: str = PROJECT, dataset: str = DATASET, table: str = "predictions"
):
    upload_location = "{}.{}.{}".format(project, dataset, table)
    logger.info("Appending data to BigQuery at {}".format(upload_location))
    client = bigquery.Client(project)
    job_config = bigquery.LoadJobConfig(
        schema=[
            bigquery.SchemaField("coolblue_cookie_id", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("visit_date", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("input_product_sequence", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("predicted_product_types", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("prediction_timestamp", bigquery.enums.SqlTypeNames.TIMESTAMP),
        ],
        write_disposition="WRITE_TRUNCATE",  # replace partition with new data
    )
    job = client.load_table_from_dataframe(df, upload_location, job_config=job_config)
    job.result()  # wait for the job to complete

    table = client.get_table(upload_location)  # Make an API request
    logger.info(
        "Added {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), upload_location
        )
    )


def downcast_numeric_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = df.select_dtypes(include=["float"])
    int_cols = df.select_dtypes(include=["int"])

    for cols in float_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast="float")
    for cols in int_cols.columns:
        df[cols] = pd.to_numeric(df[cols], downcast="integer")

    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drops exact duplicates (checks all included columns to qualify, also visit_date if present)
    Args:
        df (pd.DataFrame): Dataframe to check for duplicate rows.
    Returns:
        pd.DataFrame: Dataframe without repeating duplicates.
    """
    logger.info("     Removing exact duplicates on columns {}".format(df.columns.values))
    df_len = len(df)
    df = df.drop_duplicates(keep="last")  # data is sorted by date asc, keep newest observations
    df = df.reset_index(drop=True)
    logger.info("         Removed {} duplicate rows".format(df_len - len(df)))
    return df


def subset_train_pred_data(
    df: pd.DataFrame,
    mode: str = RUN_MODE,
    pred_days: int = N_DAYS_TO_PREDICT,
    date_column: str = "visit_date",
) -> pd.DataFrame:
    """Splits the dataframe into subsets for training and prediction.
    Args:
        df (pd.DataFrame): Dataframe to split.
        mode (str): Either experiment or production
    Returns:
        pd.DataFrame: Dataframe for training.
        pd.DataFrame: Dataframe for prediction.
    """

    if mode == "production":
        if DRY_RUN:
            logger.info("     Subsetting training and prediction df for dry run production")
            pred_rows = int(0.1 * len(df))  # predict for latest 10% of data
            df_pred = df.tail(pred_rows)
        else:
            logger.info("     Subsetting training and prediction df for production")
            latest_date = np.sort(df["visit_date"].unique())[-1]  # latest day in data
            # train on all available data, predict on latest available day
            df_pred = df[df[date_column] >= latest_date]

    elif mode == "experiment":
        if DRY_RUN:
            logger.info("     Subsetting training and prediction df for dry run experiment")
            pred_rows = int(0.1 * len(df))
            df_pred = df.iloc[pred_rows:]
            df = df.iloc[:pred_rows]
        else:
            logger.info("     Subsetting training and prediction df for experiment")
            first_pred_date = np.sort(df["visit_date"].unique())[-pred_days]  # latest 2 days
            df_pred = df[df[date_column] >= first_pred_date]  # predict on latest complete day
            df = df[df[date_column] < first_pred_date]  # remove testing from training data

    return df, df_pred


def get_date_range(df: pd.DataFrame, target_column: str = "visit_date") -> pd.DataFrame:
    """
    Retrieves the minimum and maximum date present in a column of a pd.DataFrame
    """
    min_date = df[target_column].min()
    max_date = df[target_column].max()
    return min_date, max_date


def tokenize_sequences_train(dataframe: pd.DataFrame, target_column: str = "product_sequence"):
    """Short summary.
    Args:
        dataframe (pd.DataFrame): Dataframe with sequence strings.
        target_column (str): The column with comma seperated text to tokenize.
    Returns:
        np.ndarray: Description of returned object.
        Tokenizer: Tokenizer object for product_id to product_type_id.
        Tokenizer: Tokenizer object for product_type_id to numeric tokens.
    """
    logger.info("     Tokenizing sequences for training")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=N_ITEMS)
    tokenizer.fit_on_texts(dataframe[target_column])
    sequences = tokenizer.texts_to_sequences(dataframe[target_column])
    return sequences, tokenizer


def tokenize_sequences_predict(
    dataframe: pd.DataFrame,
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    target_column: str = "product_sequence",
) -> np.ndarray:
    """Short summary.
    Args:
        dataframe (pd.DataFrame): Dataframe with sequence strings.
        tokenizer_step1 (Tokenizer): Tokenizer object for product_id to product_type_id.
        tokenizer_step2 (Tokenizer): Tokenizer object for product_type_id to numeric tokens.
        target_column (str): The column with comma seperated text to tokenize.
    Returns:
         np.ndarray: Matrix with tokenized sequences.
    """
    logger.info("     Tokenizing sequences for prediction")
    tokenized_sequences = tokenizer.texts_to_sequences(dataframe[target_column])
    return tokenized_sequences


def get_n_items_included_in_vocab(tokenizer: tf.keras.preprocessing.text.Tokenizer) -> np.int:
    # n_items = len(tokenizer.word_index.values()) + 1
    n_items = N_ITEMS + 1
    logger.info("     {} items included in the data".format(n_items))
    return n_items


def pad_sequences(array: np.ndarray) -> np.ndarray:
    """Pre-pad sequences with 0's, length is based on longest present sequence.
    Args:
        array (np.ndarray): Matrix of sequences.
    Returns:
        np.ndarray: Matrix of padded sequences.
    """
    logger.info("     Padding sequences")
    sequences = tf.keras.preprocessing.sequence.pad_sequences(array, padding="pre", dtype=np.int16)
    return sequences


def filter_valid_sequences(array: np.ndarray, min_input_items: int = MIN_ITEMS_TRAIN) -> np.ndarray:
    """Filters sequences that are not valid. Invalid sequences do not contain enough products or end
    in a 0 (padding). This can occur due to creating subsequences of longer sequences for training.
    Args:
        array (array): Matrix of sequences.
        min_input_items (int): Treshold for filtering invalid sequences, +1 due to excluding target.
    Returns:
        np.ndarray: Valid sequences.
    """
    logger.info("     Filtering valid sequences")
    min_input_items = min_input_items + 1  # correction since we need a target for training/testing!
    pre_len = len(array)
    array = array[array[:, -1] != 0]  # ending in 0 is a duplicate subsequence or empty sequence
    min_product_mask = np.sum((array != 0), axis=1) >= min_input_items
    valid_sequences = array[min_product_mask]
    logger.info("         Removed {} invalid sequences".format(pre_len - len(valid_sequences)))
    logger.info("         Kept {} valid sequences".format(len(valid_sequences)))
    return valid_sequences


def force_labels_inside_valid_interval(array: np.ndarray, max_value: int) -> np.ndarray:
    """In some cases one of the tokenization/input steps is buggy. This can lead to a labels that
    are not understood by the network resulting in a crash. So we check if the labels
    are inside the valid range for the embedding layer. If they fall outside, replace them with 0's.
    Args:
        array (np.ndarray): Matrix of tokenized sequences
    Returns:
        np.ndarray:
    """
    invalid_label_count = np.sum(array[array >= max_value])
    array[array >= max_value] = 0  # range is exclusive [min,max)
    logger.info("     Replaced {} invalid labels with padding".format(invalid_label_count))


def filter_repeated_unique_sequences(array: np.ndarray, min_input_items: int = 1) -> np.ndarray:
    """Filters sequences that do not contain enough variation in items. A sequence with repeated
    unique items does not contain much information about relations with other items. Since we filter
    before creating subsequences out of longer sequences we still retain information about which
    items are often repeated. This is due the the fact that repetition also occurs in sequences
    containing more than a n unique items.
    Args:
        array (np.ndarray): Matrix with padded sequences.
        min_input_items (int): Threshold of required unique items per sequence.
    Returns:
        np.ndarray: Matrix of filtered sequences
    """
    logger.info("     Filtering repeated unique sequences")
    pre_len = len(array)
    unique_count_per_row = np.apply_along_axis(func1d=lambda x: len(set(x)), arr=array, axis=1)
    repeated_uniques_mask = unique_count_per_row == min_input_items + 1  # +1 due to padding
    valid_sequences = array[~repeated_uniques_mask]
    logger.info(
        "         Removed {} sequences containing {} repeated unique items".format(
            pre_len - len(valid_sequences), min_input_items
        )
    )
    logger.info("         Kept {} valid sequences".format(len(valid_sequences)))
    return valid_sequences


# this numpy function is fast but a bit tricky, be sure to validate output when changing stuff!
def generate_train_test_pairs(
    array: np.ndarray, input_length: int = WINDOW_LEN, step_size: int = 1
) -> np.ndarray:
    """Creates multiple subsequences out of longer sequences in the matrix to be used for training.
    Output shape is based on the input_length. Note that the output width is input_length + 1 since
    we later take the last column of the matrix to obtain an input matrix of width input_length and
    a vector with corresponding targets (next item in that sequence).
    Args:
        array (array): Matrix with equal length padded sequences.
        input_length (int): Size of sliding window, equal to desired length of input sequence.
        step_size (int): Can be used to skip items in the sequence.
    Returns:
        array: Reshaped matrix with # columns equal to input_length + 1 (input + target item).
    """
    shape = (array.size - input_length + 2, input_length + 1)
    strides = array.strides * 2
    window = np.lib.stride_tricks.as_strided(array, strides=strides, shape=shape)[0::step_size]
    return window


def drop_batching_remainder(data, batch_size: int = BATCH_SIZE):
    """Short summary.
    Args:
        data (np.ndarray / pd.DataFrame): Matrix to be divided into equal batches.
        batch_size (int): How many rows define one single batch.
    Returns:
        np.ndarray: Matrix with a number of rows that will fit into equal batches.
    """
    skip_rows = len(data) % batch_size
    data_without_remainder = data[skip_rows:]
    logger.info("     Dropped {} rows to fit data into batches of {}".format(skip_rows, BATCH_SIZE))
    return data_without_remainder


def split_input_output(array: np.ndarray):
    """Splits off the last column from the sequence matrix to be used as target for each row.
    Args:
        array (np.ndarray): Matrix of sequences.
    Returns:
        np.ndarray: Matrix of input sequences (X)
        np.ndarray: Matrix of target items (y)
    """
    X = array[:, :-1]  # anything except last column are input items
    y = array[:, -1]  # last column is output target
    return X, y


def extract_top_items(y_pred_probs: np.ndarray, output_length: int = N_ITEMS_TO_PRED) -> np.ndarray:
    """Function to extract predicted output sequences. Output is based on the predicted logit values
    where the highest probability corresponds to the first ranked item and so forth.
    Output positions are based on probability from high to low so the output sequence is ordered.
    To be used for obtaining multiple product ranked and calculating MAP@K values.
    Args:
        y_pred_probs (np.ndarray): Predicted probabilities for all included products.
        output_length (int): Number of top K products to extract from the probability array.
    Returns:
        np.ndarray: Matrix of predicted product tokens with shape [sequences_pred, output_length]
    """
    # obtain indices of highest logit values, the position corresponds to the encoded item
    ind_of_max_logits = np.argpartition(y_pred_probs, -output_length)[-output_length:]
    # order the sequence, sorting the negative values ascending equals sorting descending
    y_pred = ind_of_max_logits[np.argsort(-y_pred_probs[ind_of_max_logits])]

    return y_pred


# @nb.njit(nb.types.Array(nb.int16, 1, "A")(nb.float32[:]))
# def numba_extract_top_items(array: np.ndarray, output_length: int = N_ITEMS_TO_PRED) -> np.ndarray:
#     """
#     Gets the indices of the top k values in an (1-D) array.
#     * NOTE: The returned indices are not sorted based on the top values.
#     """
#     sorted_indices = np.zeros((output_length,), dtype=np.float32)
#     minimum_index = 0
#     minimum_index_value = 0
#     for value in array:
#         if value > minimum_index_value:
#             sorted_indices[minimum_index] = value
#             minimum_index = sorted_indices.argmin()
#             minimum_index_value = sorted_indices[minimum_index]
#     # in some situations, due to different resolution, you get k-1 results - this is to avoid that!
#     minimum_index_value -= np.finfo(np.float32).resolution
#     return (array >= minimum_index_value).nonzero()[0][::-1][:output_length]


def predict_and_extract_top_items(sequences: np.ndarray) -> np.ndarray:
    """Predicts on a matrix of sequences and extracts the indices of the highest probabilities.
    The index position corresponds to the numeric token of the items.
    Args:
        sequences (np.ndarray): Matrix of input sequences to predict on.
    Returns:
        np.ndarray: Matrix with ordered predicted item tokens.
    """
    t0_pred = time.time()
    logger.info("     Predicting and extracting item probabilities [{} mode]".format(RUN_MODE))
    y_pred_probs = model.predict(sequences, batch_size=BATCH_SIZE)
    y_pred = np.apply_along_axis(func1d=extract_top_items, axis=1, arr=y_pred_probs)
    time_pred = time.time() - t0_pred
    logger.info(
        "[v] Elapsed time for {} items * {} sequences: {:.4} seconds".format(
            n_items, len(sequences), time_pred
        )
    )
    return y_pred


def average_precision(actual: int, prediction, k: int = 5) -> float:
    score = 0.0
    for i, pred in enumerate(prediction[:k]):
        if actual == pred:
            score = 1 / (i + 1)
            break
    return score


def mean_average_precision(actuals: int, predictions: np.ndarray, k: int = 5) -> float:
    apks = [
        average_precision(actual=actual, prediction=prediction, k=k)
        for actual, prediction in zip(actuals, predictions)
    ]
    mean_apks = np.mean(apks)
    return mean_apks


def overlap_per_sequence(X_test: np.ndarray, y_pred: np.ndarray, k: int = 5) -> int:
    """Finds overlapping items that are present in both arrays per row.
    Args:
        X_test (np.ndarray): Input sequences for testing.
        y_pred (np.ndarray): Predicted output sequences.
    Returns:
        list: A list of overlapping products for each row
    """
    overlap_items = [set(X_test[row, -k:]) & set(y_pred[row, :k]) for row in range(len(X_test))]
    return overlap_items


def mean_novelty(X_test: np.ndarray, y_pred: np.ndarray, k: int = 5) -> float:
    """Computes the average overlap over all input and predicted sequences. Note that novelty is
    computed as 1 - overlap as the new items are the ones that are not present in both arrays.
    Args:
        X_test (np.ndarray): Input sequences for testing.
        y_pred (np.ndarray): Predicted output sequences.
    Returns:
        int: Average novelty over all predicted items.
    """
    overlap_items = overlap_per_sequence(X_test, y_pred, k=k)
    overlap_sum = np.sum([len(overlap_items[row]) for row in range(len(overlap_items))])
    mean_novelty = 1 - (overlap_sum / (len(X_test) * X_test.shape[1]))  # new items do not overlap
    return mean_novelty


# ==================================================================================================
# [+] DATA LOADING
# ==================================================================================================

if LOGGING:  # and not DRY_RUN:
    mlflow.start_run()  # start mlflow run for experiment tracking
t0_input = time.time()

logger.info("[+] Loading input data")
sequence_df = pd.read_csv(
    "./neural-product-personalization/data/daily_sequences_20200211.csv", encoding="utf-8",
)
product_map_df = pd.read_csv(
    "./neural-product-personalization/data/product_map_df.csv", encoding="utf-8"
)

# product_map_df = query_product_map_data()
# if RUN_MODE == "production" and not TRAIN_MODEL:  # add intraday session data for prediction
#     sequence_df = query_product_sequence_with_intraday_data()
# else:  # only complete session days
#     sequence_df = query_product_sequence_data()

sequence_df.tail(int(3135764 * N_WEEKS_OF_DATA))  # 3 weeks of data
rows_in_raw_df = len(sequence_df)

if DRY_RUN:
    sequence_df = sequence_df.tail(300000)  # we need enough data to ensure non empty subsets

time_input = (time.time() - t0_input) / 60
logger.info("[v] Elapsed time for loading input data: {:.3} minutes".format(time_input))

# ==================================================================================================
# [+] DATA PROCESSING FOR MODEL TRAINING
# ==================================================================================================

sequence_df, sequence_pred_df = subset_train_pred_data(df=sequence_df, mode=RUN_MODE)
sequence_pred_df = drop_duplicate_rows(sequence_pred_df)  # predict on newest session @ cookie level
min_train_date, max_train_date = get_date_range(sequence_df)
min_pred_date, max_pred_date = get_date_range(sequence_pred_df)
sequence_pred_df["visit_date"] = sequence_pred_df["visit_date"].astype(str)
logger.info("     Training data ranges from {} to {}".format(min_train_date, max_train_date))
logger.info("     Predicton data ranges from {} to {}".format(min_pred_date, max_pred_date))

if not TRAIN_MODEL:  # check if we can load a trained model from GCS, if not load data and re-train
    logger.info("   Attempting to download model from GCS: {}/{}/model/".format(PROJECT, BUCKET))
    try:
        model = download_trained_model_from_gcs(PROJECT, BUCKET)
        total_params = model.count_params()
        tokenizer = download_tokenizer_from_gcs(PROJECT, BUCKET)
        n_items = get_n_items_included_in_vocab(tokenizer)
        logger.info("[v] Succesfully loaded trained model and tokenizers")
    except IOError:
        logger.info("[x] Failed to load trained model or tokenizers, setting TRAIN_MODEL to True")
        TRAIN_MODEL = True


if TRAIN_MODEL:
    t0_data = time.time()
    logger.info("[+] Processing input data for model training")

    sequences, tokenizer = tokenize_sequences_train(
        dataframe=sequence_df, target_column="product_sequence"
    )
    del sequence_df

    n_items = get_n_items_included_in_vocab(tokenizer)
    sequences = pad_sequences(sequences)
    sequences = filter_valid_sequences(sequences, min_input_items=MIN_ITEMS_TRAIN)
    force_labels_inside_valid_interval(sequences, max_value=n_items)  # np fancy inplace indexing
    if FILTER_REPEATED_UNIQUES:
        sequences = filter_repeated_unique_sequences(sequences)

    logger.info("     Reshaping into train/test subsequences with fixed window size for training")
    sequences = np.apply_along_axis(func1d=generate_train_test_pairs, axis=1, arr=sequences)
    sequences = np.vstack(sequences)
    logger.info("         Generated {} subsequences for training".format(len(sequences)))

    sequences = filter_valid_sequences(
        sequences, min_input_items=MIN_ITEMS_TRAIN
    )  # re-filter subseqs
    sequences_train, sequences_val = train_test_split(
        sequences, test_size=VAL_RATIO, shuffle=SHUFFLE_TRAIN_AND_VAL
    )
    sequences_train = drop_batching_remainder(sequences_train)
    sequences_val = drop_batching_remainder(sequences_val)
    if SHUFFLE_TRAIN:
        sequences_train = shuffle(sequences_train)
    X_train, y_train = split_input_output(sequences_train)
    X_val, y_val = split_input_output(sequences_val)

    logger.info("     Final dataset shapes:")
    logger.info("         Training X {}, y {}".format(X_train.shape, y_train.shape))
    logger.info("         Validation X {}, y {}".format(X_val.shape, y_val.shape))
    logger.info("     Final dataset memory footprints")
    logger.info(
        "         Training X {:.3}, y {:.3} MB".format(X_train.nbytes * 1e-6, y_train.nbytes * 1e-6)
    )
    logger.info(
        "         Validation X {:.3}, y {:.3} MB".format(X_val.nbytes * 1e-6, y_val.nbytes * 1e-6)
    )
    logger.info(
        "[v] Elapsed time for processing input data: {:.4} seconds".format(time.time() - t0_data)
    )
    # del sequences, sequences_train, sequences_val
    gc.collect()


# ==================================================================================================
# [+] MODEL TRAINING
# ==================================================================================================


if TRAIN_MODEL:
    t0_train = time.time()
    logger.info("[+] Defining Neural Network")

    def gru_network(
        input_dim: int = n_items,
        output_dim: int = EMBED_DIM,
        units: int = N_HIDDEN_UNITS,
        batch_size: int = BATCH_SIZE,
        dropout: float = DROPOUT,
        recurrent_dropout: float = RECURRENT_DROPOUT,
    ) -> tf.keras.Sequential:
        """Defines a RNN model with a trainable embedding input layer and GRU units.
        Args:
            vocab_size (int): Number of unique items included in the data.
            embed_dim (int): Number of embedding dimensions.
            num_units (int): Number of units for the GRU layer.
            batch_size (int): Number of subsequences used in a single pass during training.
            dropout (float): Probability of dropping a node.
            recurrent_dropout (float): Probability of dropping a hidden state during training.

        Returns:
            tensorflow.keras.Sequential: Model object

        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=n_items,
                    output_dim=EMBED_DIM,
                    batch_input_shape=[BATCH_SIZE, None],
                    mask_zero=True,
                ),
                tf.keras.layers.GRU(
                    units=N_HIDDEN_UNITS,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    dropout=DROPOUT,
                    recurrent_dropout=RECURRENT_DROPOUT,
                    recurrent_regularizer=REGULARIZER,
                    return_sequences=False,
                    unroll=False,
                    use_bias=True,
                    stateful=True,
                    recurrent_initializer="glorot_uniform",
                    reset_after=True,
                    implementation=1,
                ),
                tf.keras.layers.Dense(n_items, activation="sigmoid"),
            ]
        )
        model.compile(
            loss="sparse_categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"]
        )
        return model

    model = gru_network()
    logger.info("   Logging model configuration {}".format(model.get_config()))

    early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0, patience=3, verbose=1, restore_best_weights=True
    )
    logger.info(
        "     Training for a maximum of {} Epochs with batch size {}".format(MAX_EPOCHS, BATCH_SIZE)
    )
    model_history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping_monitor],
    )

    history_dict = model_history.history  # store model training history results
    train_loss_values = history_dict["loss"]
    train_acc_values = history_dict["accuracy"]
    val_loss_values = history_dict["val_loss"]
    val_acc_values = history_dict["val_accuracy"]
    epochs = np.arange(1, len(train_loss_values) + 1).astype(int)  # +1 due to indexing from 0

    time_train = time.time() - t0_train
    total_params = model.count_params()
    logger.info(
        "[v] Elapsed time for training network with {} parameters on {} sequences: {:.3} minutes".format(
            total_params, len(y_train), time_train / 60
        )
    )

    del y_train
    gc.collect()

    if SAVE_MODEL_TO_DISK and not DRY_RUN:  # don't save small dry_run models
        save_trained_model_to_disk()
        logger.info("[v] Succesfully saved model to disk")
    if SAVE_MODEL_TO_GCS and not DRY_RUN:
        upload_trained_model_to_gcs(PROJECT, BUCKET)
        upload_tokenizer_to_gcs(PROJECT, BUCKET)
        logger.info("[v] Succesfully uploaded model and tokenizers to GCS")

# ==================================================================================================
# [+] MODEL PREDICTIONS
# ==================================================================================================

logger.info("[+] Predicting in {} mode".format(RUN_MODE))
t0_pred = time.time()

sequences_pred = tokenize_sequences_predict(
    dataframe=sequence_pred_df, target_column="product_sequence", tokenizer=tokenizer,
)
sequences_pred = pad_sequences(sequences_pred)
sequences_pred = sequences_pred[:, -PRED_LOOKBACK:]  # amount of items to consider for prediction


if RUN_MODE == "experiment":  # no output to BigQuery for experiments
    sequences_pred = filter_valid_sequences(sequences_pred, min_input_items=1)
    sequences_pred = drop_batching_remainder(sequences_pred)
    X_test, y_test = split_input_output(sequences_pred)
    y_pred = predict_and_extract_top_items(X_test)
elif RUN_MODE == "production":
    sequences_pred = drop_batching_remainder(sequences_pred)
    sequence_pred_df = drop_batching_remainder(sequence_pred_df)
    y_pred = predict_and_extract_top_items(sequences_pred)
    # prepare dataframe with predictions for output to bq
    predicted_products = tokenizer.sequences_to_texts(y_pred)

    sequence_pred_df["predicted_products"] = predicted_products
    del predicted_products
    sequence_pred_df["prediction_timestamp"] = datetime.datetime.now()
    # replace spaces in the string with commas to match sequence format in recommendation tables
    sequence_pred_df["predicted_products"] = [
        ",".join(sequence.split(" ")) for sequence in sequence_pred_df["predicted_products"]
    ]
    # keep only latest session per coolblue_cookie_id
    sequence_pred_df.drop_duplicates(subset=["coolblue_cookie_id"], keep="last", inplace=True)
    # ensure dataframe matches BigQuery output table format
    sequence_pred_df.rename(columns={"product_sequence": "input_product_sequence"}, inplace=True)
    sequence_pred_df = sequence_pred_df[
        [
            "coolblue_cookie_id",
            "visit_date",
            "input_product_sequence",
            "predicted_products",
            "prediction_timestamp",
        ]
    ]

    if SAVE_RESULTS_TO_BQ and not DRY_RUN:
        upload_results_to_bigquery(df=sequence_pred_df)

# del sequence_pred_df
gc.collect()

time_pred = time.time() - t0_pred

# ==================================================================================================
# [+] MODEL EVALUATION
# ==================================================================================================

if EVALUATE_MODEL:
    logger.info("[+] Evaluation for {}".format(RUN_MODE))
    t0_pred = time.time()

    if RUN_MODE == "production":  # re-process the test set for evaluation of predictions
        logger.info("       Preparing data for production evaluation")
        sequences_pred = filter_valid_sequences(sequences_pred, min_input_items=1)
        sequences_pred = drop_batching_remainder(sequences_pred)
        X_test, y_test = split_input_output(sequences_pred)
        y_pred = predict_and_extract_top_items(X_test)

    if not TRAIN_MODEL:
        X_train = X_test  # X_train is not available if no model was trained

    logger.info("     Computing evaluation metrics:")
    accuracy = accuracy_score(y_test, y_pred[:, 0])
    map2 = mean_average_precision(y_test, y_pred[:, :2], k=2)
    map4 = mean_average_precision(y_test, y_pred[:, :4], k=4)
    map5 = mean_average_precision(y_test, y_pred[:, :5], k=5)
    map6 = mean_average_precision(y_test, y_pred[:, :6], k=6)
    map8 = mean_average_precision(y_test, y_pred[:, :8], k=8)
    map10 = mean_average_precision(y_test, y_pred[:, :10], k=10)
    coverage = len(np.unique(y_pred[:, :6])) / len(np.unique(X_train))
    novelty = mean_novelty(X_test[:, -6:], y_pred[:, :6])

    accuracy_views = accuracy_score(y_test, X_test[:, -1:])
    map2_views = mean_average_precision(y_test, X_test[:, -2:], k=2)
    map4_views = mean_average_precision(y_test, X_test[:, -4:], k=4)
    map6_views = mean_average_precision(y_test, X_test[:, -6:], k=6)
    map8_views = mean_average_precision(y_test, X_test[:, -8:], k=8)
    map10_views = mean_average_precision(y_test, X_test[:, -10:], k=10)
    coverage_views = len(np.unique(X_test[:, -6:])) / len(np.unique(X_train))
    novelty_views = mean_novelty(X_test[:, -6:], X_test[:, -6:])

    pop_products = np.tile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], reps=(len(y_test), 1))
    accuracy_pop = accuracy_score(y_test, pop_products[:, 0])
    map2_pop = mean_average_precision(y_test, pop_products[:, :2], k=2)
    map4_pop = mean_average_precision(y_test, pop_products[:, :4], k=4)
    map6_pop = mean_average_precision(y_test, pop_products[:, :6], k=6)
    map8_pop = mean_average_precision(y_test, pop_products[:, :8], k=8)
    map10_pop = mean_average_precision(y_test, pop_products[:, :10], k=10)
    coverage_pop = len(np.unique(pop_products[:, :6])) / len(np.unique(X_train))
    novelty_pop = mean_novelty(X_test[:, -6:], pop_products[:, :6])

    logger.info("       Neural Item Personalizer:")
    logger.info("           Accuracy @ 1   {:.4}%".format(accuracy * 100))
    logger.info("           MAP @ 2        {:.3}".format(map2))
    logger.info("           MAP @ 4        {:.3}".format(map4))
    logger.info("           MAP @ 5        {:.3}".format(map5))
    logger.info("           MAP @ 6        {:.3}".format(map6))
    logger.info("           MAP @ 8        {:.3}".format(map8))
    logger.info("           MAP @ 10        {:.3}".format(map10))
    logger.info("           Coverage       {:.4}%".format(coverage * 100))
    logger.info("           Novelty        {:.4}%".format(novelty * 100))

    logger.info("     Baseline Metrics:")
    logger.info("       Last Viewed Items:")
    logger.info("           Accuracy @ 1   {:.4}%".format(accuracy_views * 100))
    logger.info("           MAP @ 2        {:.3}".format(map2_views))
    logger.info("           MAP @ 4        {:.3}".format(map4_views))
    logger.info("           MAP @ 6        {:.3}".format(map6_views))
    logger.info("           MAP @ 8        {:.3}".format(map8_views))
    logger.info("           MAP @ 10        {:.3}".format(map10_views))
    logger.info("           Coverage       {:.4}%".format(coverage_views * 100))
    logger.info("           Novelty        {:.4}%".format(novelty_views * 100))

    logger.info("       Most Popular Items:")
    logger.info("           Accuracy @ 1   {:.4}%".format(accuracy_pop * 100))
    logger.info("           MAP @ 2        {:.3}".format(map2_pop))
    logger.info("           MAP @ 4        {:.3}".format(map4_pop))
    logger.info("           MAP @ 6        {:.3}".format(map6_pop))
    logger.info("           MAP @ 8        {:.3}".format(map8_pop))
    logger.info("           MAP @ 10        {:.3}".format(map10_pop))
    logger.info("           Coverage       {:.4}%".format(coverage_pop * 100))
    logger.info("           Novelty        {:.4}%".format(novelty_pop * 100))


# ==================================================================================================
# [+] LOG EXPERIMENT
# ==================================================================================================

if LOGGING and not DRY_RUN:
    logger.info("🧪 Logging experiment to mlflow")

    # Set tags
    mlflow.set_tags({"tf": tf.__version__, "level": PREDICTION_LEVEL})

    # Log parameters
    mlflow.log_param("mode", RUN_MODE)
    mlflow.log_param("project", PROJECT)
    mlflow.log_param("bucket", BUCKET)
    mlflow.log_param("datetime", DATETIME)
    mlflow.log_param("save_model", SAVE_MODEL_TO_GCS)
    mlflow.log_param("train_model", TRAIN_MODEL)
    mlflow.log_param("save_results_to_bq", SAVE_RESULTS_TO_BQ)
    mlflow.log_param("n_items", n_items)
    mlflow.log_param("embed_dim", EMBED_DIM)
    mlflow.log_param("n_hidden_units", N_HIDDEN_UNITS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("dropout", DROPOUT)
    mlflow.log_param("recurrent_dropout", RECURRENT_DROPOUT)
    mlflow.log_param("trainable_params", total_params)
    mlflow.log_param("optimizer", OPTIMIZER)
    mlflow.log_param("regularizer", REGULARIZER)
    mlflow.log_param("window", WINDOW_LEN)
    mlflow.log_param("pred_lookback", PRED_LOOKBACK)
    mlflow.log_param("min_items_train", MIN_ITEMS_TRAIN)
    mlflow.log_param("shuffle_train", SHUFFLE_TRAIN)
    mlflow.log_param("shuffle_train_and_val", SHUFFLE_TRAIN_AND_VAL)
    mlflow.log_param("data_imbalance_correction", DATA_IMBALANCE_CORRECTION)
    mlflow.log_param("filter_repeated_uniques", FILTER_REPEATED_UNIQUES)
    mlflow.log_param("min_train_date", min_train_date)
    mlflow.log_param("max_train_date", max_train_date)
    mlflow.log_param("n_days_to_predict", N_DAYS_TO_PREDICT)
    mlflow.log_param("min_pred_date", min_pred_date)
    mlflow.log_param("max_pred_date", max_pred_date)
    mlflow.log_param("rows_in_raw_df", rows_in_raw_df)
    mlflow.log_param("n_weeks", N_WEEKS_OF_DATA)


    # Log metrics
    if EVALUATE_MODEL:
        mlflow.log_metric("Accuracy", np.round(accuracy, 3))
        mlflow.log_metric("MAP 2", np.round(map2, 3))
        mlflow.log_metric("MAP 4", np.round(map4, 3))
        mlflow.log_metric("MAP 5", np.round(map5, 3))
        mlflow.log_metric("MAP 6", np.round(map6, 3))
        mlflow.log_metric("MAP 8", np.round(map8, 3))
        mlflow.log_metric("MAP 10", np.round(map10, 3))
        mlflow.log_metric("coverage", np.round(coverage, 3))
        mlflow.log_metric("novelty", np.round(novelty, 3))

    # Only available if a model was trained
    if TRAIN_MODEL:
        mlflow.log_param("epochs", epochs[-1])
        mlflow.log_metric("Train mins", np.round(time_train / 60), 2)
        mlflow.log_metric("Train loss", np.round(train_loss_values[-1], 4))
        mlflow.log_metric("Train acc", np.round(train_acc_values[-1], 4))
        mlflow.log_metric("Val loss", np.round(val_loss_values[-1], 4))
    mlflow.log_metric("Pred secs", np.round(time_pred))

    # Log artifacts
    mlflow.log_artifact(
        "./neural-product-personalization/personalizer.py"
    )  # log the executed code for this run
    mlflow.log_artifact(
        "./neural-product-personalization/log/logfile_{}.log".format(DATETIME)
    )  # log log ;)
    mlflow.end_run()
    if SAVE_LOGS_TO_GCS:
        upload_log_to_gcs()
        zip_and_upload_mlruns_to_gcs()

logger.info("[v] All done, total time: {:.3} minutes".format((time.time() - t0_input) / 60))
logger.info("\n\n=============================================================================\n\n")
logger.handlers = []  # kill loggers


# ==================================================================================================
# [+] MODEL VALIDATION
# ==================================================================================================

# PRODUCT CASES

input_products = tokenizer.sequences_to_texts(X_test)
input_products = [
    ",".join(sequence.split(" ")) for sequence in input_products
]
input_products = [
    sequence.split(",") for sequence in input_products
]

predicted_products = tokenizer.sequences_to_texts(y_pred)
predicted_products = [
    ",".join(sequence.split(" ")) for sequence in predicted_products
]
predicted_products = [
    sequence.split(",") for sequence in predicted_products
]

# create mapping dictionaries
product_map = dict(
    zip(product_map_df["product_id"].astype(str), product_map_df["product_name"].astype(str),)
)


input_names = []
for j in np.arange(0, len(input_products)):
    input_names.append([product_map[i] for i in input_products[j]])
output_names = []
for j in np.arange(0, len(predicted_products)):
    output_names.append([product_map[i] for i in predicted_products[j]])

case_names = list(zip(input_names, output_names))
case_names[np.random.randint(0, len(case_names))]


##########################################################################################
# PRODUCT CASES SUMMARY

import itertools
import collections
import matplotlib.pyplot as plt
import seaborn as sns



def create_input_output_count_df(X_test, y_pred, tokenizer, product_map_df):
    combinations = [
        list(itertools.product(X_test[row][np.nonzero(X_test[row])], y_pred[row]))
        for row in np.arange(len(X_test))
    ]
    combinations = filter(None, combinations)  # filter out empty lists
    combinations = np.vstack(combinations)
    combinations_counter_dict = dict(collections.Counter(map(tuple, combinations)))
    sorted_combinations_counter_dict = {
        k: v
        for k, v in sorted(
            combinations_counter_dict.items(), key=lambda item: item[1], reverse=True
        )
    }

    tuple_df = pd.DataFrame(
        (tokenizer.index_word[token], tokenizer.index_word[product_id])
        for token, product_id in sorted_combinations_counter_dict.keys()
    )
    tuple_df["count"] = sorted_combinations_counter_dict.values()
    tuple_df.columns = ["input_id", "output_id", "count"]

    id_to_name_map_dict = dict(
        zip(
            product_map_df["product_id"].astype(str),
            product_map_df["product_name"].astype(str),
        )
    )

    tuple_df["input_name"] = tuple_df["input_id"].map(id_to_name_map_dict)
    tuple_df["output_name"] = tuple_df["output_id"].map(id_to_name_map_dict)
    return tuple_df


def plot_product_relations(tuple_df, input_name="random"):
    if input_name == "random":
        input_name = tuple_df["input_name"].sample(n=1).values[0]

    input_id = tuple_df[tuple_df["input_name"] == input_name]["input_id"].iloc[0]
    input_count = tuple_df[tuple_df["input_name"] == input_name]["count"].sum()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        x="count",
        y="output_name",
        hue="output_name",
        data=tuple_df[tuple_df["input_name"] == input_name].iloc[0:10],
        palette="viridis",
        ax=ax,
        dodge=False,
    )
    plt.title(
        "Strongest I/O Relations \n Input Item: {} \n Input Product ID: {} \n Input Occurence: {}".format(
            input_name.capitalize(), input_id, input_count
        ),
        size=13,
        weight="bold",
    )
    plt.ylabel("OCCURENCE")
    plt.xlabel("OUTPUT ITEM")
    plt.legend("")
    plt.tight_layout()


tuple_df = create_input_output_count_df(X_test, y_pred, tokenizer, product_map_df)
tuple_df.iloc[50000:50020]

plot_product_relations(tuple_df)
