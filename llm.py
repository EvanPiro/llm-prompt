from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import textwrap
import warnings
import sys

warnings.filterwarnings('ignore')

def pretty_print(s):
    print("Output:\n" + 80 * '-')
    print(textwrap.fill(tokenizer.decode(s, skip_special_tokens=True), 80))


##################################################
## instantiating LLM & its tokenizer
##################################################

# model_to_use = "gpt2"
model_to_use = "gpt2-large"

print("Using model: ", model_to_use)

# get the tokenizer for the pre-trained LM you would like to use
tokenizer = GPT2TokenizerFast.from_pretrained(model_to_use)

# instantiate a model (causal LM)
model = GPT2LMHeadModel.from_pretrained(model_to_use,
                                        output_scores=True,
                                        pad_token_id=tokenizer.eos_token_id)

# inspecting the (default) model configuration
# (it is possible to created models with different configurations)
print(model.config)

prompt = sys.argv[1]

# translate the prompt into tokens
input_tokens = tokenizer(prompt, return_tensors="pt").input_ids
print(input_tokens)

outputs = model.generate(input_tokens,
                         max_new_tokens=100,
                         do_sample=True,
                         top_k=50,
                         )

print("\nTop-k sampling:\n")
pretty_print(outputs[0])
