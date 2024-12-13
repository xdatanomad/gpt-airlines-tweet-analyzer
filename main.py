
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
    num_tokens = num_tokens_from_string("A config file is required to run the application. Please create a config file called `config.yaml` in the current directory.")
    print(f"Number of tokens: {num_tokens}")


if __name__ == '__main__':
    main()
