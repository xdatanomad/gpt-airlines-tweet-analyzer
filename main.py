import os
import sys
import logging
import logging.handlers
import yaml
import json
import time
from datetime import datetime
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_incrementing
from scipy import spatial
import click
from openai import OpenAI


# ========================================
# SETTING UP LOGGING AND APPLICATION CONFIGURATION
#
# Application configuration is loaded from a config file called `config.yaml`.
# The configuration file is required to run the application.
# This file should be placed in the current directory.
# This file contains many parameters such as model names, prompts, etc.
# ========================================

# Path to the default config file
DEFAULT_CONFIG_FILE = "config.yaml"

# Setup logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Sets up logging for the application.

    This function configures a logger with both console and file handlers. The console handler logs messages to the console, 
    while the file handler logs messages to a file with rotation based on size.

    Args:
        log_level (str): The logging level for the console handler. Defaults to "INFO". 
                         Valid levels are "DEBUG", "INFO", "WARNING", "ERROR", and "CRITICAL".

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        logger = setup_logging("DEBUG")
        logger.info("This is an info message")
        logger.debug("This is a debug message")
    """
    logger = logging.getLogger('main')
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File handler
    max_logfile_size = 30 * 1024 * 1024         # MBs
    file_handler = logging.handlers.RotatingFileHandler(
        'app.log', 
        maxBytes=max_logfile_size, 
        backupCount=0
        )
    file_handler.setLevel(logging.DEBUG)        # the file logger will log everything (at DEBUG level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# Read YAML configuration file
def load_config(
        config_file: str = DEFAULT_CONFIG_FILE, 
        default_configs: dict = {}
        ) -> dict:
    """
    Loads configuration from a specified YAML file and updates the default configurations.
    Args:
        config_file (str): The path to the configuration file. Defaults to DEFAULT_CONFIG_FILE.
        default_configs (dict): A dictionary containing default configurations. Defaults to an empty dictionary.
    Returns:
        dict: The updated configuration dictionary.
    Raises:
        FileNotFoundError: If the specified configuration file is not found.
        Exception: If there is an error reading the configuration file.
    """
    try:
        with open(config_file, "r") as file:
            default_configs.update(yaml.safe_load(file))
            logger.info(f"Loaded configuration from {config_file}")
    except FileNotFoundError as e:
        logger.fatal(f"Config file {config_file} not found.")
        logger.info("A config file is required to run the application. Please create a config file called `config.yaml` in the current directory.")
        raise FileNotFoundError(f"Config file {config_file} not found.")
    except Exception as e:
        logger.fatal(f"Error reading config file {config_file}.")
        logger.warning("Exiting application.")
        raise e
    return default_configs


# setup logging and configuration
logger: logging.Logger = setup_logging("INFO")
config: dict = load_config(DEFAULT_CONFIG_FILE)

# Other useful default configurations
DEFAULT_MODEL = "gpt-3.5-turbo"                                                 # Default model name
DEFAULT_TEMPRATURE = 0.0                                                        # Default temperature
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"                              # Default embedding model

DEFAULT_TRAINING_FILEPATH = "data/airline_train.csv"                            # The provided training file. Used by the few-shot prompting method
DEFAULT_TRAINING_EMBEDDINGS_FILEPATH = "data/airline_embeddings.parquet"        # Computed embeddings for the training file. Used by the RAG method
DEFAULT_FINE_TUNING_FILEPATH = "data/airline_fine_tune.jsonl"                   # Model Fine-tuning file. Used by the fine-tuning method


# ========================================
# Application Run Statistics
#
# A great helper class to track and log the statistics of an application's run, 
# including OpenAI API call statistics and token usage.
# ========================================

class ApplicationRunStatistics:
    """
    A class to track and log the statistics of an application's run, including OpenAI API call statistics and token usage.
    Attributes:
        run_id (str): A unique identifier for the run, generated based on the current datetime.
        start_time (float): The start time of the run.
        end_time (float): The end time of the run.
        elapsed_time (float): The total elapsed time of the run.
        action (str): The action being performed during the run.
        description (str): A description of the run.
        cmdline_args (list): The command line arguments passed to the script.
        tokens_usage (dict): A dictionary to track the usage of tokens for prompt, response, embeddings, and total.
        openai_calls (dict): A dictionary to track the number and runtime of OpenAI API calls for chat completions and embeddings.
    Methods:
        update_chat_completion_stats(resp, runtime=0, log=False):
            Updates the statistics for chat completion API calls, including token usage and runtime.
        update_embeddings_stats(resp, runtime=0, log=False):
            Updates the statistics for embeddings API calls, including token usage and runtime.
        check_results():
            Placeholder method to check results (implementation not provided).
        print_stats():
            Logs the statistics of the run, including run ID, action, elapsed time, OpenAI API call statistics, and token usage.
        __str__():
            Returns a string representation of the run statistics.
    """

    def __init__(
            self,
            action: str = "Generic",            # metadata column
            description: str = "",              # metadata column
            ):
        # generate a short id for the run
        self.run_id = f"run-{datetime.now().strftime("%Y%m%d-%H%M%S")}"
        self.start_time = time.time()
        self.end_time = None
        self.elapsed_time = 0
        self.action = action
        self.description = description
        # capture command line arguments
        self.cmdline_args = sys.argv[1:]
        # tokens uages
        self.tokens_usage = {
            "prompt": 0,
            "response": 0,
            "embeddings": 0,
            "total": 0,
        }
        # openai calls stats
        self.openai_calls = {
            "chat_completion": 0,
            "embeddings": 0,
            "chat_completion_runtime": 0,
            "embeddings_runtime": 0,
        }
        # accuracy stats
        self.acuracy = 0
        self.correct_lines = 0
        self.incorrect_lines = 0
        self.nrows = 0

    def update_chat_completion_stats(self, resp, runtime: float = 0, log: bool = False):
        """
        Updates the statistics for chat completion calls to the OpenAI API.
        Args:
            resp: The response object from the OpenAI API call, which contains token usage information.
            runtime (float, optional): The runtime of the chat completion call in seconds. Defaults to 0.
            log (bool, optional): If True, logs the updated statistics. Defaults to False.
        Raises:
            Exception: Catches and passes any exception that occurs during the update process.
        """
        try:
            self.openai_calls["chat_completion"] += 1
            self.openai_calls["chat_completion_runtime"] += runtime
            self.tokens_usage["prompt"] += resp.usage.prompt_tokens
            self.tokens_usage["response"] += resp.usage.completion_tokens
            self.tokens_usage["total"] += resp.usage.total_tokens
            if log:
                logger.debug(f"ChatCompletion Stats: runtime={runtime:.3f}s, prompt_tokens={resp.usage.prompt_tokens}, response_tokens={resp.usage.completion_tokens}, total_tokens={resp.usage.total_tokens}")
        except Exception as e:
            pass

    def update_embeddings_stats(self, resp, runtime: float = 0, log: bool = False):
        """
        Updates the statistics related to embeddings based on the response from the OpenAI API.
        Args:
            resp: The response object from the OpenAI API containing usage information.
            runtime (float, optional): The time taken to get the embeddings. Defaults to 0.
            log (bool, optional): If True, logs the embeddings statistics. Defaults to False.
        Raises:
            Exception: Catches all exceptions to prevent the function from failing.
        """
        try:
            self.openai_calls["embeddings"] += 1
            self.openai_calls["embeddings_runtime"] += runtime
            self.tokens_usage["embeddings"] += resp.usage.total_tokens
            self.tokens_usage["total"] += resp.usage.total_tokens
            if log:
                logger.debug(f"Embeddings Stats: runtime={runtime:.3f}s, total_tokens={resp.usage.total_tokens}")
        except Exception as e:
            pass

    def check_results(
            self,
            df: pd.DataFrame,
            correct_col: str = "airlines",
            processed_col: str = "airlines_mentioned",
            ):
        """
        Calculates the accuracy of the extracted airlines by comparing them with the correct airlines in the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the tweets and the extracted airlines.
            correct_col (str, optional): The column name in the DataFrame containing the correct airlines. Defaults to "airlines".
            processed_col (str, optional): The column name in the DataFrame containing the extracted airlines. Defaults to "airlines_mentioned".
        """
        # check if the correct and processed columns exist in the dataframe
        if correct_col not in df.columns or processed_col not in df.columns:
            logger.warning(f"Columns {correct_col} or {processed_col} not found in the DataFrame. Accuracy is NOT calculated.")
            return
        # calculate the accuracy by comparing the correct airlines with the extracted airlines
        correct = df[correct_col].apply(set)
        processed = df[processed_col].apply(set)
        accuracy = (correct == processed).mean()
        # calculate number of correct and incorrect lines
        correct_lines = (correct == processed).sum()
        incorrect_lines = len(df) - correct_lines
        # set stats
        self.acuracy = accuracy
        self.correct_lines = correct_lines
        self.incorrect_lines = incorrect_lines
        self.nrows = len(df)

    def print_stats(self, filepath: str = None):
        """
        Prints the statistics of the current run including run ID, action, elapsed time,
        OpenAI API calls, and token usage. Optionally, the statistics can be saved to a file.
        """
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Action: {self.action}")
        # logger.info(f"Description: {self.description}")
        logger.info(f"Run Time: {self.elapsed_time:.3f}s")
        logger.info(f"OpenAI API Calls:")
        logger.info(f"  Chat Completions: {self.openai_calls['chat_completion']}")
        logger.info(f"  Chat Completions Runtime: {self.openai_calls['chat_completion_runtime']:.3f}s")
        logger.info(f"  Embeddings: {self.openai_calls['embeddings']}")
        logger.info(f"  Embeddings Runtime: {self.openai_calls['embeddings_runtime']:.3f}s")
        logger.info(f"Token Usage:")
        logger.info(f"  Prompt Tokens: {self.tokens_usage['prompt']}")
        logger.info(f"  Response Tokens: {self.tokens_usage['response']}")
        logger.info(f"  Embeddings Tokens: {self.tokens_usage['embeddings']}")
        logger.info(f"  Total Tokens: {self.tokens_usage['total']}")
        logger.info(f"Accuracy Scores:")
        logger.info(f"  Accuracy: {(self.acuracy * 100):.3f}%")
        logger.info(f"  Correct Lines: {self.correct_lines}")
        logger.info(f"  Incorrect Lines: {self.incorrect_lines}")
        # if a file path is provided, save the stats to the file
        if filepath is not None:
            header = not os.path.exists(filepath)
            record = {
                "run_id": self.run_id,
                "action": self.action,
                "nrows": self.nrows,
                "run_time": self.elapsed_time,
                "chat_completions_calls": self.openai_calls["chat_completion"],
                "chat_completion_runtime": self.openai_calls["chat_completion_runtime"],
                "embeddings_calls": self.openai_calls["embeddings"],
                "embeddings_runtime": self.openai_calls["embeddings_runtime"],
                "prompt_tokens": self.tokens_usage["prompt"],
                "response_tokens": self.tokens_usage["response"],
                "embeddings_tokens": self.tokens_usage["embeddings"],
                "total_tokens": self.tokens_usage["total"],
                "accuracy": self.acuracy,
                "correct_lines": self.correct_lines,
                "incorrect_Lines": self.incorrect_lines,
            }
            pd.DataFrame([record]).to_csv(filepath, index=False, mode="a", header=header)

    def __str__(self):
        return f"Run ID: {self.run_id}, Action: {self.action}, Elapsed Time: {self.elapsed_time:.3f}s"
        

# Define a global object to store application run statistics
job_run = ApplicationRunStatistics()



# ========================================
# OpenAI Base API Calls
#
# These functions are the base OpenAI API calls used by all other functions.
# This is the best place to change main parameters like model name, temperature, etc.
# ========================================

# check if OPENAI_API_KEY is set
if "OPENAI_API_KEY" not in os.environ:
    logger.error("OPENAI_API_KEY environment variable not set.")
    logger.info("Please set the OPENAI_API_KEY environment variable. For more information, see: https://platform.openai.com/docs/quickstart")
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# setup openai client
client = OpenAI()


# Retry (using tenacity) if call fails due to rate limiting
@retry(stop=stop_after_attempt(3), wait=wait_incrementing(start=0.6, increment=0.6, max=3))
def chat_completion(
        system_message: str,                    # system message
        prompt: str,                            # user message
        model: str = None,                      # model name
        temperature: float = None,              # temperature parameter
        ) -> dict:
    """
    Generates a chat completion response using the OpenAI API.

    Args:
        system_message (str): The system message to set the context for the chat.
        prompt (str): The user message or prompt to generate a response for.
        model (str, optional): The model name to use for the chat completion. Defaults to gpt-3.5-turbo.
        temperature (float, optional): The temperature parameter to control the randomness of the response. Defaults to 0.

    Returns:
        dict: The response from the OpenAI API in JSON format.

    Raises:
        json.JSONDecodeError: If the response from the API is not valid JSON.
        Exception: For any other errors that occur during the API call.
    """
    try:
        # Get the model configiration
        if model is None:
            model = config.get("openai", {}).get("model", DEFAULT_MODEL)
        if temperature is None:
            temperature = config.get("openai", {}).get("temperature", DEFAULT_TEMPRATURE)
        top_p = config.get("openai", {}).get("top_p", 0.0)
        frequency_penalty = config.get("openai", {}).get("frequency_penalty", 0.0)
        presence_penalty = config.get("openai", {}).get("presence_penalty", 0.0)
        # Set model output format to JSON
        response_format = {"type": "json_object"}
        # Time the call
        start_time = time.time()
        # Call chat completion API
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},      # system message. e.g. "You are an expert in identifying airlines mentioned in tweets."
                {"role": "user", "content": prompt}                 # main prompt message. e.g. "What airlines are mentioned in this tweet: {tweet}"
            ],
            temperature=temperature,                    # temperature parameter: 0.0 to 1.0. Lower values are more deterministic
            response_format=response_format,            # response format: json_object, json, or plain_text. We always use json_object. Use json for newer models. (gpt-4-turbo)
            top_p=top_p,                                # top_p parameter: 0.0 to 1.0. Lower values are more deterministic
            frequency_penalty=frequency_penalty,        # frequency_penalty parameter: 0.0 to 1.0. Lower values penalize frequent tokens
            presence_penalty=presence_penalty,          # presence_penalty parameter: 0.0 to 1.0. Lower values penalize tokens that are already present in the prompt
        )
        end_time = time.time()
        # update stats
        job_run.update_chat_completion_stats(resp, runtime=(end_time - start_time), log=True)
        # convert the response to a json
        resp = resp.choices[0].message.content
        resp = json.loads(resp)
        return resp
    except json.JSONDecodeError as e:
        # Error due to invalid JSON response
        # Remember: OpenAI API responses are NOT guaranteed to be valid JSON in gpt-3.5-turbo
        # Remember: Prompts are retried up to 3 times using tenacity if they fail
        logger.warning(f"GPT response: {resp.choices[0].message.content}")
        logger.error(f"Invalid GPT JSON response. Error parsing JSON output: {e}")
        raise e
    except Exception as e:
        # Any other error
        # Remember: Prompts are retried up to 3 times using tenacity if they fail
        logger.error(f"Error getting chat completion: {e}")
        raise e


# Retry (using tenacity) if call fails due to rate limiting
@retry(stop=stop_after_attempt(3), wait=wait_incrementing(start=0.6, increment=0.6, max=3))
def get_embeddings(
        text: str,                              # text to get embeddings for
        ) -> list:
    """
    Generate embeddings for a given text using the specified model by config file.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        list: A list representing the embeddings of the input text.
    """
    # Time the call
    start_time = time.time()
    # Get the encoding for the specified model
    embedding_model = config.get("openai", {}).get("embedding_model", DEFAULT_EMBEDDING_MODEL)
    resp = client.embeddings.create(
        model=embedding_model,
        input=text,
    )
    end_time = time.time()
    # update stats
    job_run.update_embeddings_stats(resp, runtime=(end_time - start_time), log=True)
    return resp.data[0].embedding


# ========================================
# Utility Functions
#
# These functions are utility functions used by other functions.
# ========================================

def post_process_json_response(response: dict) -> list:
    """
    Processes a JSON response to extract a list of airlines. Its main purpose is to handle different JSON response formats form the model.

    This function handles different formats of the JSON response:
    - If the response is already a list, it returns the list.
    - If the response is a dictionary, it checks for known keys ("airlines" or "airlines_mentioned")
      that contain a list and returns the corresponding list.
    - If none of the known keys are found, it returns the first key that holds a list.
    - If no list is found, it logs a warning and raises a ValueError.

    Args:
        response (dict): The JSON response to process.

    Returns:
        list: A list of airlines extracted from the response.

    Raises:
        ValueError: If the response does not contain a list of airlines.
    """
    # if the response if already a list, return it
    if isinstance(response, list):
        return response
    # if the response is a dictionary, check if it has common known keys
    elif isinstance(response, dict):
        if "airlines" in response and isinstance(response["airlines"], list):
            return response["airlines"]
        elif "airlines_mentioned" in response and isinstance(response["airlines_mentioned"], list):
            return response["airlines_mentioned"]
        else:
            # return the first key that hold a list
            for k, v in response.items():
                if isinstance(v, list):
                    return v
    # all else fails
    logger.warning(f"JSON response: {response}")
    raise ValueError("Invalid response JSON format. No list of airlines found.")


def read_tweets_to_dataframe(filepath: str = None) -> pd.DataFrame:
    """
    Reads tweets from a CSV file into a pandas DataFrame.
    Args:
        filepath (str, optional): The path to the CSV file containing the tweets. 
                                  If not provided, the path is retrieved from the config.
    Returns:
        pd.DataFrame: A DataFrame containing the tweets and their associated airlines.
    Raises:
        ValueError: If the 'airlines' column is not parsed as a list of strings.
    Notes:
        - The 'airlines' column is parsed as a list.
        - Bad lines and encoding errors in the CSV file are skipped.
        - The DataFrame is expected to have 'tweet' and 'airlines' columns.
    """
    # get the training file name form the config if not set
    if filepath is None:
        filepath = config.get("files", {}).get("training_file", DEFAULT_TRAINING_FILEPATH)
    # read the training file CSV
    # please note:
    #   - airlines column is parsed as a list
    #   - badlines and encoding errors are skipped
    df = pd.read_csv(
        filepath,
        skip_blank_lines=True,
        skipinitialspace=True,
        encoding_errors='ignore',                   # accounting for utf encoding errors
        on_bad_lines='skip',                        # skip bad tweets and lines
        usecols=["tweet", "airlines"],              # ensure required columns are present
        converters={"airlines": eval},              # parse the airlines column as a list
        )
    # check if all airlines were correctly parsed as a list
    if not all([isinstance(x, list) for x in df["airlines"]]):
        logger.error(f"airlines column is not a list of strings")
        raise ValueError(f"airlines column is not a list of strings")
    logger.info(f"Loaded training set from file: {filepath}")
    # return the training dataframe
    return df


def read_tweets_embeddings_to_dataframe(filepath: str = None) -> pd.DataFrame:
    """
    Reads tweet embeddings from a parquet file and loads them into a pandas DataFrame.
    Args:
    filepath (str, optional): The path to the training embeddings file. If not provided, 
                    the path will be retrieved from the configuration.
    
    Returns:
    pd.DataFrame: A DataFrame containing the tweet embeddings.
    
    aises:
    FileNotFoundError: If the specified embeddings file does not exist.
    
    Notes:
        - If the `filepath` is not provided, it will be fetched from the configuration using the key 
            "training_embeddings_file".
        - The embeddings file is expected to be in parquet format.
    """
    # get the training file name form the config if not set
    if filepath is None:
        filepath = config.get("files", {}).get("training_embeddings_file", DEFAULT_TRAINING_EMBEDDINGS_FILEPATH)
    # if the file doesn't exist, throw a message with instructions to compute the embeddings
    if not os.path.exists(filepath):
        logger.error(f"Training embeddings file not found: {filepath}")
        logger.info(f"Please run the `compute_and_save_embeddings_for_training_set` function to compute the embeddings.")
        raise FileNotFoundError(f"Training embeddings file not found: {filepath}")
    # read the embeddings parquet file
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded training set with embeddings from file: {filepath}")
    # return the training dataframe with embeddings
    return df


def embeddings_rag_search(
        tweet: str,                                 # tweet to get similiar tweets for
        training_df: pd.DataFrame,                  # training dataframe with embeddings
        nrows: int = 5,                             # number of similiar tweets to return
        embeddings_col: str = "tweet_embeddings",   # column name for the embeddings
        ) -> pd.DataFrame:
    """
    Find similar tweets based on embeddings using cosine similarity. This method is used in the RAG method.

    Args:
    tweet (str): The tweet to find similar tweets for.
    training_df (pd.DataFrame): The training dataframe containing tweet embeddings.
    nrows (int, optional): The number of similar tweets to return. Defaults to 5.
    embeddings_col (str, optional): The column name for the embeddings in the dataframe. Defaults to "tweet_embeddings".

    Returns:
    pd.DataFrame: A dataframe containing the top `nrows` similar tweets based on cosine similarity.

    Raises:
    Exception: If there is an error in calculating the similarities or processing the embeddings.
    """
    try:
        # get the embeddings for the tweet
        tweet_embeddings = get_embeddings(tweet)
        # filter out the embeddings column with None or empty list of embeddings
        training_df = training_df[training_df[embeddings_col].notnull()]
        training_df = training_df[training_df[embeddings_col].apply(lambda x: len(x) > 0)]
        # calculate the cosine similarity between the tweet and all the tweets in the training set
        training_df["similarity"] = training_df[embeddings_col].map(lambda x: 1 - spatial.distance.cosine(tweet_embeddings, x))
        # get the top num_tweets similiar tweets
        return training_df.nlargest(nrows, "similarity")
    except Exception as e:
        logger.error(f"Error getting similiar tweets from embeddings: {e}")
        raise e


# ========================================
# Main Application Actions
#
# These functions are the main application functions:
#   - run_zero_shot: run the zero-shot method on the test set
#   - run_few_shot: run the few-shot method on the test set
#   - run_rag: run the RAG method on the test set
#   - run_fine_tuning: run the fine-tuning method on the test set
#   - build_rag_embeddings_db: compute and save embeddings used in RAG method
#   - submit_fine_tuning_model_job: submit a model fine-tuning job used in fine-tuning method
# ========================================

def build_rag_embeddings_db(
        training_file: str = None,                     # path to the training file
        ) -> None:
    """
    Computes and caches embeddings for the training set.
    This function reads a training file containing tweets, computes embeddings
    for each tweet, and saves the embeddings to a parquet file for future use.
    
    Args:
    training_file (str, optional): Path to the training file containing tweets.
    """
    # log start message
    logger.info('-' * 80)
    logger.info("Computing embeddings for the training set.")
    logger.info("This is a one-time operation. The embeddings will be cached for future use.")
    logger.info("This will take a while! Go grab a coffee :)")

    # load the training file
    df = read_tweets_to_dataframe(training_file)
    # calculate embeddings for the tweets column
    df["tweet_embeddings"] = df["tweet"].map(get_embeddings)
    # save the embeddings to a parquet file
    embeddings_file = config.get("files", {}).get("training_cache", DEFAULT_TRAINING_EMBEDDINGS_FILEPATH)
    df.to_parquet(embeddings_file)
    logger.info(f"Embeddings saved to file: {embeddings_file}")


def run_zero_shot(
        df: pd.DataFrame,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        ):
    """
    Processes a DataFrame of tweets to identify mentioned airlines using a zero-shot learning approach.

    Notes:
        - The model is prompted with a system message and the tweet text read from the config file.
        - You can tweak the promtps in the config file to improve the accuracy of the model.
        - Edit the config for: prompts.zero_shot
        - Since this function uses the zero-shot learning approach to identify airlines mentioned in tweets, it has low accuracy.

    Args:
        df (pd.DataFrame): The DataFrame containing tweets.
        tweet_col (str, optional): The column name in the DataFrame that contains the tweets. Defaults to "tweet".
        airlines_col (str, optional): The column name in the DataFrame where the identified airlines will be stored. Defaults to "airlines_mentioned".

    Returns:
        pd.DataFrame: The DataFrame with an additional column containing the identified airlines for each tweet.
    """
    # setup the result col
    df[airlines_col] = None
    for i, row in df.iterrows():
        try:
            logger.info(f"line: {i + 1} tweet: {row[tweet_col]}")
            # grab the right prompt from the config
            system_message = config["prompts"]["zero_shot"]["system"]
            prompt = config["prompts"]["zero_shot"]["prompt"]
            # insert the tweet into the prompt
            prompt = prompt.format(tweet=row[tweet_col])
            # call the completion function
            json_resp = chat_completion(system_message, prompt)
            # parse the response
            airlines = post_process_json_response(json_resp)
            # add the airlines to the dataframe
            df.at[i, airlines_col] = airlines
            # check if the airlines are correct
        except Exception as e:
            logger.error(f"Error extracting airline names from tweet: {row[tweet_col]}")
            logger.error(f"Error: {e}")
    return df


def run_few_shot(
        df: pd.DataFrame,
        num_examples: int = 10,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        ):
    """
    Processes a DataFrame of tweets to extract mentioned airlines using few-shot prompting.

    Notes:
        - This method provides a static set of examples from the training set to the model for few-shot prompting.
        - The set if picked at random and is controled by the `num_examples` parameter.
        - The model is prompted with a system message, the tweet text, and a few examples from the training set.
        - You can tweak the prompts in the config file to improve the accuracy of the model.
        - Edit the config for: prompts.few_shot
        - Since this function uses a few-shot prompting approach to identify airlines mentioned in tweets, it has higher accuracy than zero-shot learning.

    Args:
        df (pd.DataFrame): DataFrame containing tweets to process.
        num_examples (int, optional): Number of examples to use for few-shot prompting. Defaults to 10.
        tweet_col (str, optional): Column name in the DataFrame containing the tweets. Defaults to "tweet".
        airlines_col (str, optional): Column name in the DataFrame to store the extracted airlines. Defaults to "airlines_mentioned".

    Returns:
        pd.DataFrame: DataFrame with an additional column containing the extracted airlines for each tweet.
    """
    # load the training set
    training_df = read_tweets_to_dataframe()
    # get a few _static_ examples from few-shot prompting
    fewshot_examples = training_df.sample(num_examples)
    fewshot_examples_str = fewshot_examples.to_csv(index=False)
    # setup the result col
    df[airlines_col] = None
    for i, row in df.iterrows():
        try:
            logger.info(f"line: {i + 1} tweet: {row[tweet_col]}")
            # grab the right prompt from the config
            system_message = config["prompts"]["few_shot"]["system"]
            prompt = config["prompts"]["few_shot"]["prompt"]
            # insert prompt and few-shot examples into the prompt
            prompt = prompt.format(tweet=row[tweet_col], examples=fewshot_examples_str)
            # call the completion function
            json_resp = chat_completion(system_message, prompt)
            # parse the response
            airlines = post_process_json_response(json_resp)
            # add the airlines to the dataframe
            df.at[i, airlines_col] = airlines
        except Exception as e:
            logger.error(f"Error extracting airline names from tweet: {row[tweet_col]}")
            logger.error(f"Error: {e}")
    return df


def run_rag(
        df: pd.DataFrame,
        num_examples: int = 5,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        ):
    """
    Processes a DataFrame of tweets to identify mentioned airlines using a Retrieval-Augmented Generation (RAG) approach.

    Notes:
        - This method uses a RAG model to generate responses based on a combination of the tweet text and similar examples from the training set.
        - The similar examples are retrieved based on cosine similarity of embeddings.
        - The number of similar examples to retrieve is controlled by the `num_examples` parameter.
        - The model shares the same prompts as the few-shot prompting method.
        - Edit the config for: prompts.few_shot
        - Since this function uses a RAG model with similar examples, it has higher accuracy than few-shot prompting.
    Args:
        df (pd.DataFrame): DataFrame containing tweets to be processed.
        num_examples (int, optional): Number of similar examples to retrieve for RAG. Defaults to 5.
        tweet_col (str, optional): Column name in the DataFrame containing the tweet text. Defaults to "tweet".
        airlines_col (str, optional): Column name in the DataFrame to store the identified airlines. Defaults to "airlines_mentioned".

    Returns:
        pd.DataFrame: DataFrame with an additional column containing the identified airlines for each tweet.
    """
    # load the training set
    embeddings_df = read_tweets_embeddings_to_dataframe()
    df[airlines_col] = None
    for i, row in df.iterrows():
        try:
            logger.info(f"line: {i + 1} tweet: {row[tweet_col]}")
            # crab the right prompt from the config
            system_message = config["prompts"]["few_shot"]["system"]
            prompt = config["prompts"]["few_shot"]["prompt"]
            # find smiliar tweets from a cosine similarity search on embeddings
            sim_tweets = embeddings_rag_search(row[tweet_col], embeddings_df, nrows=num_examples)
            examples_csv = sim_tweets[["tweet", "airlines"]].to_csv(index=False)
            # update the prompt with tweet and RAG based examples
            prompt = prompt.format(tweet=row[tweet_col], examples=examples_csv)
            # call the completion function
            json_resp = chat_completion(system_message, prompt)
            # parse the response
            airlines = post_process_json_response(json_resp)
            # add the airlines to the dataframe
            df.at[i, airlines_col] = airlines
        except Exception as e:
            logger.error(f"Error extracting airline names from tweet: {row[tweet_col]}")
            logger.error(f"Error: {e}")
    return df


def run_fine_tuning(
        df: pd.DataFrame,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        ):
    """
    Uses a pre-trained model on a given DataFrame of tweets to extract mentioned airlines.

    Notes:
        - This method uses a fine-tuned model to extract airlines mentioned in tweets.
        - The fine-tuned model is expected to be pre-trained by the submit_fine_tuning_model_job() method.
        - The model is prompted with a system message and the tweet text.
        - Edit the config for: prompts.fine_tuning
        - Since this function uses a fine-tuned model, it has higher accuracy than the other methods (but runs a little slower).

    Args:
        df (pd.DataFrame): DataFrame containing tweets.
        tweet_col (str, optional): Column name in the DataFrame containing the tweets. Defaults to "tweet".
        airlines_col (str, optional): Column name in the DataFrame to store the extracted airlines. Defaults to "airlines_mentioned".

    Raises:
        FileNotFoundError: If the fine-tuned model is not found in the configuration.

    Returns:
        pd.DataFrame: DataFrame with an additional column containing the extracted airlines.
    """
    # load the fine-tuned model
    fine_tuned_model = config.get("openai", {}).get("pre_fine_tuned_model", None)
    # check if the fine-tuned model exists
    if fine_tuned_model is None:
        logger.error("Fine-tuned model not found.")
        logger.info("Please run the `fine_tune_model_jobrun` function to fine-tune a model.")
        raise FileNotFoundError("Fine-tuned model not found.")
    # 
    df[airlines_col] = None
    for i, row in df.iterrows():
        try:
            logger.info(f"line: {i + 1} tweet: {row[tweet_col]}")
            # call the completion function
            system_message = config["prompts"]["fine_tuning"]["system"]
            prompt = config["prompts"]["fine_tuning"]["prompt"]
            prompt = prompt.format(tweet=row[tweet_col])
            json_resp = chat_completion(system_message, prompt, model=fine_tuned_model)
            # parse the response
            airlines = post_process_json_response(json_resp)
            # add the airlines to the dataframe
            df.at[i, airlines_col] = airlines
        except Exception as e:
            logger.error(f"Error extracting airline names from tweet: {row[tweet_col]}")
            logger.error(f"Error: {e}")
    # return the dataframe
    return df


def submit_fine_tuning_model_job():
    """
    Submits a fine-tuning job for a GPT model based on a training set.
    This function performs the following steps:
        1. Logs the start of the fine-tuning process.
        2. Loads the training set from a specified CSV file.
        3. Generates an NDJSON file for fine-tuning.
        4. Uploads the fine-tuning file to OpenAI.
        5. Submits a fine-tuning job using the uploaded file.
        6. Waits for the fine-tuning job to complete.
        7. Logs the completion of the fine-tuning job and the name of the fine-tuned model.
    Raises:
        Exception: If any step in the fine-tuning process fails, an exception is logged and re-raised.
    Note:
        Ensure that the YAML config file is updated with the new fine-tuned model name after the job completes.
    """
    # from a gpt model based on the training set
    # load the training set
    try:
        # start log message
        logger.info('-' * 80)
        logger.info("Starting fine-tuning process.")
        logger.info('-' * 80)
        logger.info("This will take a few minutes to complete.")

        # load the training set file
        training_file = config.get("files", {}).get("training_file", DEFAULT_TRAINING_FILEPATH)
        logger.info(f"Loading training set from file: {training_file}")
        df = pd.read_csv(
            training_file,
            skip_blank_lines=True,
            skipinitialspace=True,
            encoding_errors='ignore',                   # accounting for utf encoding errors
            on_bad_lines='skip',                        # skip bad tweets and lines
            usecols=["tweet", "airlines"],              # ensure required columns are present
            converters={"airlines": eval},              # parse the airlines column as a list
            )
        
        # generate an ndjson file for training
        ft_filename = config.get("files", {}).get("fine_tuning_file", DEFAULT_FINE_TUNING_FILEPATH)
        logger.info(f"Generating fine-tuning ndjson file: {ft_filename}")
        with open(ft_filename, "w") as file:
            for i, row in df.iterrows():
                line = {
                    "messages": [
                        {"role": "system", "content": "You are an expert in identifying airlines mentioned in tweets."},
                        {"role": "user", "content": f"What airlines are mentioned in this tweet: {row['tweet']}"},
                        {"role": "assistant", "content": f"{row['airlines']}"},
                    ],
                }
                file.write(json.dumps(line) + "\n")

        # upload the fine-tuning file to openai
        logger.info(f"Uploading fine-tuning file to OpenAI: {ft_filename}")
        file_obj = client.files.create(
            file=open(ft_filename, "rb"), 
            purpose="fine-tune"
            )
        logger.info(f"Fine-tuning file uploaded: {file_obj.id}")

        # run a fine-tuning job
        logger.info(f"Submitting a fine-tuning job on file: {ft_filename}")
        ft_model = config.get("openai", {}).get("fine_tuning_model", "gpt-3.5-turbo-0125")
        job = client.fine_tuning.jobs.create(
            model=ft_model,
            training_file=file_obj.id,
        )
        logger.info(f"Fine-tuning job started: {job.id}")

        # wait for the job to complete
        while job.status != "succeeded":
            time.sleep(5)
            job = client.fine_tuning.jobs.retrieve(job.id)
            logger.info(f"Checking fine-tuning job status: {job.status}")
        fine_tuned_model = job.fine_tuned_model
        logger.info(f"Fine-tuning job completed. Model name: {fine_tuned_model}")
        logger.info(f"PLEASE update the YAML config file with the new fine-tuned model name: {fine_tuned_model}")
    except Exception as e:
        logger.error(f"Fine-tuning job failed! Error: {e}")
        raise e


# ========================================
# Main Function
#
# Main command line function to run the tweet analysis methods.
#
# Example usage:
#   python main.py --run fine_tuning
#   python main.py --run few_shot --num-examples 20
#   python main.py --run zero_shot --input-file data/airlines_test.csv --config-file config.yaml
# ========================================

@click.command()
@click.option('--run', type=click.Choice(['zero_shot', 'few_shot', 'rag', 'fine_tuning', 'setup_rag', 'setup_finetuning'], case_sensitive=False), required=True, help='Action to perform')
@click.option('--input-file', type=click.Path(exists=True), help='Path to the input file. If omitted airlines_test.csv file is used.', required=False)
@click.option('--config-file', type=click.Path(exists=True), help='Path to YAML config file. If omitted config.yaml is used.', required=False)
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], case_sensitive=False), default='INFO', help='Application log level')
@click.option('--num-examples', type=int, default=10, help='Number of examples to be used with few-shot and RAG methods')
def main(run, input_file, config_file, log_level, num_examples, **kwargs):
    """
    Main function to run various tweet analysis methods based on the provided command line arguments.
    Args:
        run (str): The action to perform. Options include 'zero_shot', 'few_shot', 'rag', 'fine_tuning', 'setup_rag', 'setup_finetuning'.
        input_file (str): Path to the input file containing tweets.
        config_file (str): Path to the configuration file.
        log_level (str): Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        num_examples (int): Number of examples to use for few-shot and RAG methods.
        **kwargs: Additional keyword arguments.
    Returns:
        None
    """

    # setup logging level from the command line
    logger.handlers[0].setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # load the config file if provided and update the default settings
    if config_file:
        global config
        config = load_config(config_file, config)
    
    # setting up input file to process
    if input_file:
        logger.info(f"Using input file: {input_file}")
    else:
        input_file = config.get("files", {}).get("test_file", None)
        logger.info(f"Using default input file: {input_file}")
    if input_file is None or not os.path.exists(input_file):
        logger.error("Input file is required but either missing or does not exist. Exiting.")
        return
    
    # run the appropriate method based on the command line
    job_run.action = run    # log the action taken in this job
    if run == 'zero_shot':
        df = read_tweets_to_dataframe(filepath=input_file)
        df = run_zero_shot(df)
    elif run == 'few_shot':
        df = read_tweets_to_dataframe(filepath=input_file)
        df = run_few_shot(df, num_examples)
    elif run == 'rag':
        df = read_tweets_to_dataframe(filepath=input_file)
        df = run_rag(df, num_examples)
    elif run == 'fine_tuning':
        df = read_tweets_to_dataframe(filepath=input_file)
        df = run_fine_tuning(df)
    elif run == 'setup_rag':
        build_rag_embeddings_db(training_file=input_file)
        logger.info("Embeddings database built.")
        return
    elif run == 'setup_finetuning':
        submit_fine_tuning_model_job()
        logger.info("Fine-tuning job finished.")
        return
    # check results and print stats
    logger.info("\nResults:\n")
    print(df)
    output_file = config.get("files", {}).get("output_file", "data/airline_output.csv")
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to file: {output_file}")
    # computer accuracy scores and print the stats
    job_run.check_results(df)
    job_run.print_stats(filepath="stats.csv")


# ========================================
# Unit Tests
#
# These functions are only used for testing purposes.
# ========================================

def test_similirity_search():
    # load the training set
    embeddings_df = read_tweets_embeddings_to_dataframe()
    print(embeddings_df)
    tweet = r"US Airways i tried it but doesnt help very much and Reservation seems to be overwhelmed with some issues"
    similar_tweets = embeddings_rag_search(tweet, embeddings_df)
    print(similar_tweets)


def test_fine_tune_model_jobrun():
    submit_fine_tuning_model_job()


def test_zero_shot_example():
    # load the test file
    test_file = config.get("files", {}).get("test_file", None)
    df = read_tweets_to_dataframe(filepath=test_file)
    # sample a few rows
    df = df.sample(10)
    # run the zero-shot example
    df = run_zero_shot(df)
    # print the results
    print(df)
    job_run.print_stats()


def test_few_shot_example():
    # load the test file
    test_file = config.get("files", {}).get("test_file", None)
    df = read_tweets_to_dataframe(filepath=test_file)
    # sample a few rows
    df = df.sample(10)
    # run the few-shot example
    num_examples = 20
    df = run_few_shot(df, num_examples)
    # print the results
    print(df)
    job_run.print_stats()



def test_rag_example():
    # load the test file
    test_file = config.get("files", {}).get("test_file", None)
    df = read_tweets_to_dataframe(filepath=test_file)
    # sample a few rows
    df = df.sample(10)
    # run the few-shot example
    num_examples = 10
    df = run_rag(df, num_examples)
    # print the results
    print(df)
    job_run.print_stats()


def test_finetuned_example():
    # load the test file
    test_file = config.get("files", {}).get("test_file", None)
    df = read_tweets_to_dataframe(filepath=test_file)
    # sample a few rows
    df = df.sample(10)
    # run the few-shot example
    df = run_fine_tuning(df)
    # print the results
    print(df)
    job_run.print_stats()


# ========================================
# Main Application Entry Point
# ========================================

if __name__ == '__main__':
    main()
    # test_similirity_search()
    # test_fine_tune_model_jobrun()
    # test_zero_shot_example()
    # test_few_shot_example()
    # test_rag_example()
    # test_finetuned_example()
