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



class ApplicationRunStatistics:
    def __init__(
            self,
            action: str = "Generic",
            ):
        # generate a short id for the run
        self.run_id = f"run-{datetime.now().strftime("%Y%m%d-%H%M%S")}"
        self.start_time = time.time()
        self.end_time = None
        self.action = action
        # capture command line arguments
        self.cmdline_args = sys.argv[1:]
        # tokens uages
        self.tokens_uages = {
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

    def update_chat_completion_stats(self, resp, runtime: float = 0, log: bool = False):
        try:
            self.openai_calls["chat_completion"] += 1
            self.openai_calls["chat_completion_runtime"] += runtime
            self.tokens_uages["prompt"] += resp.usage.prompt_tokens
            self.tokens_uages["response"] += resp.usage.completion_tokens
            self.tokens_uages["total"] += resp.usage.total_tokens
            if log:
                logger.debug(f"ChatCompletion Stats: runtime={runtime:.3f}s, prompt_tokens={resp.usage.prompt_tokens}, response_tokens={resp.usage.completion_tokens}, total_tokens={resp.usage.total_tokens}")
        except Exception as e:
            pass

    def update_embeddings_stats(self, resp, runtime: float = 0, log: bool = False):
        try:
            self.openai_calls["embeddings"] += 1
            self.openai_calls["embeddings_runtime"] += runtime
            self.openai_calls["embeddings"] += resp.usage.total_tokens
            if log:
                logger.debug(f"Embeddings Stats: runtime={runtime:.3f}s, total_tokens={resp.usage.total_tokens}")
        except Exception as e:
            pass

    def check_results():
        pass
        

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


def load_training_df(
        training_file: str = None,                     # path to the training file
        ) -> pd.DataFrame:
    
    # get the training file name form the config if not set
    if training_file is None:
        training_file = config.get("files", {}).get("training_file", DEFAULT_TRAINING_FILEPATH)
    # read the training file CSV
    # please note:
    #   - airlines column is parsed as a list
    #   - badlines and encoding errors are skipped
    df = pd.read_csv(
        training_file,
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
    logger.info(f"Loaded training set from file: {training_file}")
    # return the training dataframe
    return df


def load_training_embeddings_df(
        training_embeddings_file: str = None,                     # path to the training file
        ) -> pd.DataFrame:
    
    # get the training file name form the config if not set
    if training_embeddings_file is None:
        training_embeddings_file = config.get("files", {}).get("training_embeddings_file", DEFAULT_TRAINING_EMBEDDINGS_FILEPATH)
    # if the file doesn't exist, throw a message with instructions to compute the embeddings
    if not os.path.exists(training_embeddings_file):
        logger.error(f"Training embeddings file not found: {training_embeddings_file}")
        logger.info(f"Please run the `compute_and_save_embeddings_for_training_set` function to compute the embeddings.")
        raise FileNotFoundError(f"Training embeddings file not found: {training_embeddings_file}")
    # read the embeddings parquet file
    df = pd.read_parquet(training_embeddings_file)
    logger.info(f"Loaded training set with embeddings from file: {training_embeddings_file}")
    # return the training dataframe with embeddings
    return df


def embeddings_rag_search(
        tweet: str,                                 # tweet to get similiar tweets for
        training_df: pd.DataFrame,                  # training dataframe with embeddings
        nrows: int = 5,                             # number of similiar tweets to return
        embeddings_col: str = "tweet_embeddings",   # column name for the embeddings
        ) -> pd.DataFrame:
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
# ========================================

def compute_and_save_embeddings_for_training_set(
        training_file: str = None,                     # path to the training file
        ) -> None:
    
    # log start message
    logger.info('-' * 80)
    logger.info("Computing embeddings for the training set.")
    logger.info("This is a one-time operation. The embeddings will be cached for future use.")
    logger.info("This will take a while! Go grab a coffee :)")

    # load the training file
    df = load_training_df(training_file)
    # calculate embeddings for the tweets column
    df["tweet_embeddings"] = df["tweet"].map(get_embeddings)
    # save the embeddings to a parquet file
    embeddings_file = config.get("files", {}).get("training_cache", DEFAULT_TRAINING_EMBEDDINGS_FILEPATH)
    df.to_parquet(embeddings_file)
    logger.info(f"Embeddings saved to file: {embeddings_file}")


def zero_shot_example(
        df: pd.DataFrame,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        ):
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


def few_shot_example(
        df: pd.DataFrame,
        num_examples: int = 10,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        ):
    # load the training set
    training_df = load_training_df()
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


def rag_example(
        df: pd.DataFrame,
        num_examples: int = 5,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        ):
    # load the training set
    embeddings_df = load_training_embeddings_df
    df[airlines_col] = None
    df["correct"] = False
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


def finetuning_example(
        df: pd.DataFrame,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        ):
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


def fine_tune_model_jobrun():
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
        logger.info(f"Fine-tuning job completed. Model name: {job.fine_tuned_model}")
    except Exception as e:
        logger.error(f"Fine-tuning job failed! Error: {e}")
        raise e



def main():
    pass


# ========================================
# Unit Tests
#
# These functions are only used for testing purposes.
# ========================================

def test_similirity_search():
    # load the training set
    embeddings_df = load_training_embeddings_df()
    print(embeddings_df)
    tweet = r"US Airways i tried it but doesnt help very much and Reservation seems to be overwhelmed with some issues"
    similar_tweets = embeddings_rag_search(tweet, embeddings_df)
    print(similar_tweets)


def test_fine_tune_model_jobrun():
    fine_tune_model_jobrun()


def test_zero_shot_example():
    # load the test file
    test_file = config.get("files", {}).get("test_file", None)
    df = load_training_df(training_file=test_file)
    # sample a few rows
    df = df.sample(10)
    # run the zero-shot example
    df = zero_shot_example(df)
    # print the results
    print(df)


def test_few_shot_example():
    # load the test file
    test_file = config.get("files", {}).get("test_file", None)
    df = load_training_df(training_file=test_file)
    # sample a few rows
    df = df.sample(10)
    # run the few-shot example
    num_examples = 20
    df = few_shot_example(df, num_examples)
    # print the results
    print(df)


def test_rag_example():
    # load the test file
    test_file = config.get("files", {}).get("test_file", None)
    df = load_training_df(training_file=test_file)
    # sample a few rows
    df = df.sample(10)
    # run the few-shot example
    num_examples = 10
    df = rag_example(df, num_examples)
    # print the results
    print(df)


def test_finetuned_example():
    # load the test file
    test_file = config.get("files", {}).get("test_file", None)
    df = load_training_df(training_file=test_file)
    # sample a few rows
    df = df.sample(10)
    # run the few-shot example
    df = finetuning_example(df)
    # print the results
    print(df)


# ========================================
# Main Application Entry Point
# ========================================

if __name__ == '__main__':
    # main()
    test_similirity_search()
    # test_fine_tune_model_jobrun()
    # test_finetuned_example()
    # test_zero_shot_example()
    # test_few_shot_example()
    # test_rag_example()
