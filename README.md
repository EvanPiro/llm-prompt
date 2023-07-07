# LLM Prompt

## Get set up
Ensure you have python3.11 installed. Run `./setup.sh`.

## Run a Prompt
```commandline
python llm.py "The prompt you want to submit"
```
After waiting a bit (around 2 minutes for me) you should see the prompt expanded by the model into some insane robot speak.

## To Dos
- [ ] Set up a commandline interface for specifying a local path to a CSV dataset, and train the model with it before running the prompt.
- [ ] Get question and answer structure instead of next-word guessing. 

