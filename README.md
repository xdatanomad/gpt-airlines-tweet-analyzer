# Code and Setup Instructions for Extracting Airline Names from Tweets

As promised, I’ve put together a solution that extracts airline names from tweets, along with several methods to boost accuracy. The attached main.py script demonstrates four approaches:
	1.	Zero-shot Prompting:
Uses GPT-3.5-turbo directly (no examples) to identify airlines. It’s fast but less accurate.
	2.	Few-shot Prompting:
Provides a handful of examples from the training set to improve accuracy. More tokens are used, but results are better than zero-shot.
	3.	RAG + Few-shot Prompting:
Dynamically chooses examples similar to the incoming tweet (via embeddings), improving accuracy even more. Requires pre-computing and storing embeddings but yields great results.
	4.	Fine-tuned Model:
The most accurate approach. We train a custom model on the entire dataset. Once fine-tuned, it’s highly reliable but involves extra steps and time upfront.

###  Execuive Summary of the Results

Results at a Glance (on the test set `airlines_test.csv`):

- Zero-shot prompting: accuracy=76.67%, total tokens=71386, runtime=154 seconds
- Few-shot prompting: accuracy=84.00%, total tokens=175459, runtime=159 seconds
- RAG + Few-shot prompting: accuracy~=94.00%, -- currently takes long to execute
- Fine-tuned model: accuracy=98.33%, total tokens=23598, runtime=275 seconds


## Getting Started
	
### 1. Download Files:
Place main.py, requirements.txt, and config.yaml in a project directory:

```bash
mkdir ~/gpt-airline-extractor
cd ~/gpt-airline-extractor
# move the downloaded files here
```

### 2. Data Files:
Copy airlines_train.csv and airlines_test.csv into a data folder:

```bash
mkdir data
mv ~/Downloads/airlines_train.csv ~/Downloads/airlines_test.csv data/
```

### 3. Set Up Python:
Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. OpenAI API Key:
Set your OpenAI API key:

```bash
export OPENAI_API_KEY="your-key-here"
```
(Add this line to your .bashrc or .zshrc for persistence.)

## Running the Program

Try out different methods:

```bash
python main.py --run zero_shot
python main.py --run few_shot --input-file data/airlines_test.csv --num-examples 10
python main.py --run rag --input-file data/airlines_test.csv --num-examples 7
python main.py --run fine_tuning --nrows 50
```

The program is designed to run different methods. You can `--run` with the following options: 
`zero_shot`, `few_shot`, `rag`, `fine_tuning`, `setup_rag`, `setup_finetuning`.

**Notes:**
- Use --run to select the method.
- The `setup_rag` and `setup_finetuning` are one-time operations to pre-calculate the embeddings and fine-tune a model. 
- You can run these once before running rag or fine-tuning, respectively.
- Since I've already pre-trained a model, you can probably get away with NOT running `setup_finetuning` to use `fine_tunning` method.
- `--input-file` controls the input file to process. The default is `data/airlines_test.csv`.
- `--num-examples` controls the number of examples to use for few-shot prompting and RAG. The default is 10.
- `--nrows` runs the program on a random subset of tweets instead of the entire file. It's great for testing.


## Reviewing Results
- `stats.csv` holds detailed summary stats of each run (accuracy, tokens used, runtime).
- It's your best friend to compare the results of each method.
- The final processed tweets and identified airlines are in data/airline_output.csv.
- Check app.log for detailed logs.

## Tweak & Optimize

config.yaml contains model names, prompts, and other parameters. You can experiment with different models (e.g., GPT-4) and prompts for better accuracy or performance.

## Next Steps and Optimizations

An easy improvement is to use newer models like GPT-4-turbo. This would likely improve accuracy and provides much better JSON support.

As you have noticed, the RAG methods take a very long time to execute. This is mainly due to how the `get_embeddings` function is implemented. It makes an API call for each single tweet. This can be optimized by batching the embeddings calls to the API. This would significantly reduce the time. Sorry, didn't have time to implement this BUT happy to look at it together if needed.

Other optimization is to combine the RAG and fine-tuning methods -- best of both worlds, accuracy and efficiency.

## Conclusion

I hope this helps you get started. Please let me know if you have any questions or need help with anything.

In conclusion:
- fine_tuning is the most accurate AND takes least number of tokens... but takes a bit longer to run and finetuning tokens cost a bit more which is something to consider.
- RAG is the next best thing. It's accurate while using moderate amount of tokens... but we need to batch the embeddings calls to make it more efficient.

## tl;dr: Going over the Code

Program flow is fairly simple: read the tweets using pandas -> process the tweets using the selected method -> save the results to a CSV file.

- The code should be well documented and easy to follow/extend.
- `main()` function is the entry point to the program. It parses the command line arguments and runs the selected method.
- The `run_zero_shot()`, `run_few_shot()`, `run_rag()`, and `run_fine_tuning()` functions are the main methods that run the different techniques.
- The `submit_fine_tuning_model_job()` function submits a fine-tuning job to the OpenAI API to train a model on the training set.
- The `build_rag_embeddings_db()` function computes and caches embeddings for the training set. This is used in the RAG method.
- The `get_embeddings()` function is used to get embeddings for a given text (single tweet). This is used in the RAG method.
- The `embeddings_rag_search()` function finds similar tweets based on embeddings using cosine similarity. This is used in the RAG method.
