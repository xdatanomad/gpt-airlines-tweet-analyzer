import logging.handlers
import logging
import yaml
import json
import pandas as pd
from openai import OpenAI
from tiktoken import encoding_for_model
from codetiming import Timer
import csv


DEFAULT_CONFIG_FILE = "config.yaml"


logger: logging.Logger = None
config: dict = {}

# Set up logging
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



# setup openai client
client = OpenAI()


def chat_completion(
        system_message: str,                    # system message
        prompt: str,                            # user message
        ) -> dict:
    try:
        # Get the completion from the API
        global client
        model = config.get("openai", {}).get("model", "gpt-3.5-turbo")
        # get openai chat completion fine-tuning parameters
        temperature = config.get("openai", {}).get("temperature", 0.0)
        top_p = config.get("openai", {}).get("top_p", 0.0)
        frequency_penalty = config.get("openai", {}).get("frequency_penalty", 0.0)
        presence_penalty = config.get("openai", {}).get("presence_penalty", 0.0)
        response_format = {"type": "json_object"}

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            response_format=response_format,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        # print the response token uasge
        # logger.debug(f"promopt:\n{prompt}")
        logger.info(f"ChatCompletion::TokenUsage: prompt={completion.usage.prompt_tokens}, response={completion.usage.completion_tokens}, total={completion.usage.total_tokens}")
        resp = completion.choices[0].message.content
        # convert the response to a json
        resp = json.loads(resp)
        return resp
    except json.JSONDecodeError as e:
        logger.warning(f"GPT response: {completion.choices[0].message.content}")
        logger.error(f"Invalid GPT JSON response. Error parsing JSON output: {e}")
        raise e
    except Exception as e:
        logger.error(f"Error getting chat completion: {e}")
        raise e


def parse_response_json_for_list_of_airlines(response: dict) -> list:
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


def airline_name_extractor_zeroshot(
        df: pd.DataFrame,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        test_col: str = "airlines",
        ):
    df[airlines_col] = None
    df["correct"] = False
    for i, row in df.iterrows():
        try:
            logger.info(f"line: {i + 1} tweet: {row[tweet_col]}")
            # call the completion function
            system_message = config["prompts"]["zero_shot"]["system"]
            prompt = config["prompts"]["zero_shot"]["prompt"]
            # df[airlines_col] = df[tweet_col].map(lambda x: )
            prompt = prompt.format(tweet=row[tweet_col])
            json_resp = chat_completion(system_message, prompt)
            # parse the response
            airlines = parse_response_json_for_list_of_airlines(json_resp)
            # add the airlines to the dataframe
            df.at[i, airlines_col] = airlines
            # check if the airlines are correct
            if eval(f"list({row[test_col]})") == airlines:
                df.at[i, "correct"] = True
            # if i > 25:
            #     break
        except Exception as e:
            logger.error(f"Error extracting airline names from tweet: {row[tweet_col]}")
            logger.error(f"Error: {e}")
    return df



def load_training_set(
        training_file: str,                     # path to the training file
        ) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            training_file,
            skip_blank_lines=True,
            skipinitialspace=True,
            encoding_errors='ignore',
            on_bad_lines='skip',
            usecols=["tweet", "airlines"],
            converters={"airlines": eval},
            )
        # check the columns to make sure we have the required columns
        # cols = ["tweet", "airlines"]
        # if not all([col in df.columns for col in cols]):
        #     logger.error(f"Missing required columns in training file: {cols}")
        #     raise ValueError(f"Missing required columns in training file: {cols}")
        # parse the airlines column as a list
        # df["airlines"] = df["airlines
        # check if airlines column is actually a list of stirngs
        if not all([isinstance(x, list) for x in df["airlines"]]):
            logger.error(f"airlines column is not a list of strings")
            raise ValueError(f"airlines column is not a list of strings")
        return df
    except Exception as e:
        logger.error(f"Error loading training set from file: {training_file}")
        raise e


def few_shot_airline_name_extractor(
        df: pd.DataFrame,
        num_examples: int = 10,
        tweet_col: str = "tweet",
        airlines_col: str = "airlines_mentioned",
        test_col: str = "airlines",
        ):
    # load the training set
    training_file = config["files"]["training_file"]
    training_df = load_training_set(training_file)
    # get a few examples from the training set
    fewshot_df = training_df.sample(num_examples)
    fewshot_csv_string = fewshot_df.to_csv(index=False, header=True, quoting=csv.QUOTE_MINIMAL)
    df[airlines_col] = None
    df["correct"] = False
    for i, row in df.iterrows():
        try:
            logger.info(f"line: {i + 1} tweet: {row[tweet_col]}")
            # call the completion function
            system_message = config["prompts"]["few_shot"]["system"]
            prompt = config["prompts"]["few_shot"]["prompt"]
            # df[airlines_col] = df[tweet_col].map(lambda x: )
            prompt = prompt.format(tweet=row[tweet_col], examples=fewshot_csv_string)
            json_resp = chat_completion(system_message, prompt)
            # parse the response
            airlines = parse_response_json_for_list_of_airlines(json_resp)
            # add the airlines to the dataframe
            df.at[i, airlines_col] = airlines
            # check if the airlines are correct
            if eval(f"list({row[test_col]})") == airlines:
                df.at[i, "correct"] = True
            if i > 50:
                break
        except Exception as e:
            logger.error(f"Error extracting airline names from tweet: {row[tweet_col]}")
            logger.error(f"Error: {e}")
    return df



def chunk_dataframe_by_max_tokens(df: pd.DataFrame, max_tokens=12000, tokens_col: str = "tokens"):
    """returns the next chunk of the dataframe until a max number of tokens are reached"""
    start_index = 0
    end_index = 0
    token_sum = 0
    for i, row in df.iterrows():
        token_sum += row[tokens_col]
        if token_sum > max_tokens:
            end_index = i - 1
            yield df.iloc[start_index:end_index]
            start_index = end_index
            token_sum = row[tokens_col]
    # yield the last chunk
    if start_index < len(df):
        yield df.iloc[start_index:]


def build_airline_names_mapping_dict_from_training_set(
        training_file: str,                     # path to the training file
        ) -> dict:
    try:
        df = pd.read_csv(
            training_file,
            skip_blank_lines=True,
            skipinitialspace=True,
            encoding_errors='ignore',
            on_bad_lines='skip',
            )
        # check the columns to make sure we have the required columns
        cols = ["tweet", "airlines"]
        if not all([col in df.columns for col in cols]):
            logger.error(f"Missing required columns in training file: {cols}")
            raise ValueError(f"Missing required columns in training file: {cols}")
        # parse the airlines column as a list
        df["airlines"] = df["airlines"].apply(lambda x: eval(f"list({x})"))
        # compute the number of tokens
        compute_tokens_for_dataframe(df, cols=cols)
        
        # data
        print(f"df shape: {df.shape}")
        airline_aliases = {}
        total_rows = 0
        for chunk in chunk_dataframe_by_max_tokens(df):
            print(f"chuck size: {chunk.shape}, tokens: {chunk['tokens'].sum()}, mix/max index: {chunk.index.min()}/{chunk.index.max()}")
            # chunk['str_col'] = chunk.apply(lambda x: x.to_dict(), axis=1)
            total_rows += len(chunk)
            # reset the index
            chunk.reset_index(drop=True, inplace=True)
            # output the chunk to a csv formatted string buffer
            csv_buffer = chunk[cols].to_csv(index=False, header=True, quoting=csv.QUOTE_STRINGS)
            # print(csv_buffer)
            # call the completion function
            # call the prompt completion function
            system_message = config["prompts"]["airline_name_mapping"]["system"]
            prompt = config["prompts"]["airline_name_mapping"]["prompt"]
            top_p = config["prompts"]["airline_name_mapping"]["top_p"]
            frequency_penalty = config["prompts"]["airline_name_mapping"]["frequency_penalty"]
            presence_penalty = config["prompts"]["airline_name_mapping"]["presence_penalty"]
            prompt = prompt.format(data=csv_buffer)
            print(prompt)
            completion = chat_completion(system_message, prompt, top_p=top_p, frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
            print("\n\n\n")
            print(completion)
            print("\n\n\n")
            for k, v in completion.items():
                if k not in airline_aliases:
                    airline_aliases[k] = v
                else:
                    airline_aliases[k].extend(v)
        # make sure the values are unique
        for k, v in airline_aliases.items():
            airline_aliases[k] = list(set(v))
        print(f"total rows: {total_rows}")
        print("\n\n\n")
        print(airline_aliases)
        print("\n\n\n")

        # # post process the airline names
        # system = config["prompts"]["airline_name_mapping_post_process"]["system"]
        # prompt = config["prompts"]["airline_name_mapping_post_process"]["prompt"]
        # # insert data
        # prompt = prompt.format(data=airline_aliases)
        # completion = chat_completion(system, prompt)
        # # save the airline names into the yaml config file
        # data = {"airlines_mapping": completion}
        data = {"airlines_mapping": airline_aliases}
        with open("airlines_mapping.yml", "w") as file:
            yaml.dump(data, file)
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
    # df = pd.DataFrame({
    #     "text": ["A config file is required to run the application. Please create a config file called `config.yaml` in the current directory.", "This is a test string."]
    # })
    # compute_tokens_for_dataframe(df, "text")
    # print(df)

    filepath = r"data/airline_test.csv"
    df = pd.read_csv(
            filepath,
            skip_blank_lines=True,
            skipinitialspace=True,
            encoding_errors='ignore',
            on_bad_lines='skip',
            )
    # rdf = airline_name_extractor_zeroshot(df)
    # rdf.to_csv("data/airlines_zeroshot.csv", index=False)
    rdf = few_shot_airline_name_extractor(df)
    rdf.to_csv("data/airlines_fewshot.csv", index=False)


if __name__ == '__main__':
    main()
