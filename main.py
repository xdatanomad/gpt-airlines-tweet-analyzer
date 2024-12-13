
import logging.handlers
import os
import logging
import yaml
import pandas as pd
from openai import OpenAI
from tiktoken import encoding_for_model
from codetiming import Timer



DEFAULT_CONFIG_FILE = "config.yaml"


logger: logging.Logger = None
config: dict = {}

# Set up logging
def setup_logging(log_level: str = "INFO"):
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
    max_logfile_size = 10 * 1024 * 1024         # 10 MB
    file_handler = logging.handlers.RotatingFileHandler(
        'app.log', 
        maxBytes=max_logfile_size, 
        backupCount=0
        )
    file_handler.setLevel(logging.DEBUG)        # the file logger will log everything (at DEBUG level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# set up module logger
logger: logging.Logger = setup_logging("DEBUG")


# read config file
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


# setup application configuration
config = load_config(DEFAULT_CONFIG_FILE, config)


@Timer(name="tokenizer", text="Metrics::Tokenizer: secs={:.3f}", logger=logger.debug)
def num_tokens_from_string(string: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        # Get the encoding for the specified model
        encoding = encoding_for_model(model)
        # Encode the string and return the number of tokens
        return len(encoding.encode(string))
    except Exception as e:
        logger.error(f"Error getting number of tokens from string: {e}")
        logger.error(f"Moost likely the model {model} is not supported.")
        raise e


def compute_tokens_for_dataframe(
        df: pd.DataFrame,                       # dataframe to compute number of tokens for
        cols: str | list[str],                  # list of columns to compute number of tokens for
        token_col: str = "tokens",              # name of the column to store the number of tokens
        model: str = "gpt-3.5-turbo"            # model to use for tokenization
        ) -> pd.Series:
    try:
        # Get the encoding for the specified model
        encoding = encoding_for_model(model)
        # make sure cols is a lits
        if isinstance(cols, str):
            cols = [cols]
        # Encode the string and return the number of tokens
        tmp = df[cols].apply(lambda x: str(x.to_dict()), axis=1)
        tmp = tmp.map(lambda x: len(encoding.encode(x)))
        if token_col is not None:
            df[token_col] = tmp
        # Encode the string and return the number of tokens
        return tmp
    except Exception as e:
        logger.error(f"Error getting number of tokens from dataframe: {e}")
        logger.error(f"Moost likely the model {model} is not supported.")
        raise e


def build_airline_names_mapping_dict_from_training_set(
        training_file: str,                     # path to the training file
        airline_col: str = "airline",           # name of the column with airline names
        airline_id_col: str = "airline_id"       # name of the column with airline ids
        ) -> dict:
    try:
        # Get the unique airline names
        airline_names = training_set[airline_col].unique()
        # Get the unique airline ids
        airline_ids = training_set[airline_id_col].unique()
        # Create a dictionary mapping airline names to airline ids
        airline_names_mapping = dict(zip(airline_names, airline_ids))
        return airline_names_mapping
    except Exception as e:
        logger.error(f"Error building airline names mapping dictionary from training set: {e}")
        raise e


def main():
    # Create an instance of the OpenAI class
    client = OpenAI()

    # Get the completion from the API
    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "What is the purpose of life?"}
    #     ]
    # )

    # # Print the completion
    # print(completion.__dict__, "\n\n")
    # print(completion.choices[0].message.content)

    # Get the number of tokens in a string
    # num_tokens = num_tokens_from_string("A config file is required to run the application. Please create a config file called `config.yaml` in the current directory.")
    # print(f"Number of tokens: {num_tokens}")

    # Get the number of tokens in a dataframe
    df = pd.DataFrame({
        "text": ["A config file is required to run the application. Please create a config file called `config.yaml` in the current directory.", "This is a test string."]
    })
    compute_tokens_for_dataframe(df, "text")
    print(df)


if __name__ == '__main__':
    main()
