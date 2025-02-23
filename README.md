# Llama Reborn
Acknowledgement: This project is based on the Carnegie Mellon University assignment(based on a previous [minbert-assignment](https://github.com/neubig/minbert-assignment))

This is an attempt at developing a minimalist version of Llama2, part of Carnegie Mellon University's [CS11-711 Advanced NLP](https://cmu-l3.github.io/anlp-spring2025/).

In this project, I have attempted to implement some important components of the Llama2 model to better understanding its architecture, and perform text generation on a custom prompt, as well as sentence classification on ``sst`` dataset and ``cfimdb`` dataset with this model.

## Details

The code to implement can be found in `llama.py`, `classifier.py`, `optimizer.py` and `rope.py`. You are reponsible for writing _core components_ of Llama2 (one of the leading open source language models). In doing so, you will gain a strong understanding of neural language modeling. We will load pretrained weights for your language model from `stories42M.pt`; an 8-layer, 42M parameter language model pretrained on the [TinyStories](https://arxiv.org/abs/2305.07759) dataset (a dataset of machine-generated children's stories). This model is small enough that it can be trained (slowly) without a GPU. You are encouraged to use Colab or a personal GPU machine (e.g. a Macbook) to be able to iterate more quickly.

Once you have implemented these components, you will test our your model in 3 settings:
1) Generate a text completion (starting with the sentence `"I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is"`). You should see coherent, grammatical English being generated (though the content and topicality of the completion may be absurd, since this LM was pretrained exclusively on children's stories).
2) Perform zero-shot, prompt-based sentiment analysis on two datasets (SST-5 and CFIMDB). This will give bad results (roughly equal to choosing a random target class).
3) Perform task-specific finetuning of your Llama2 model, after implementing a classification head in `classifier.py`. This will give much stronger classification results.
4) If you've done #1-3 well, you will get an A! However, since you've come this far, try implementing something new on top of your hand-written language modeling system! If your method provides strong empirical improvements or demonstrates exceptional creativity, you'll get an A+ on this assignment.

### Important Notes
* Follow `setup.sh` to properly setup the environment, install dependencies and model weights.
* There is a detailed description of the code structure in [structure.md](./structure.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, no other external libraries are allowed (e.g., `transformers`).
* The `data/cfimdb-test.txt` file provided to you does **not** contain gold-labels, and contains a placeholder negative (-1) label. Evaluating your code against this set will show lower accuracies so do not worry if the numbers don't make sense.
* We will run your code with commands below (under "Reference outputs/accuracies"), so make sure that whatever your best results are reproducible using these commands.
    * Do not change any of the existing command options (including defaults) or add any new required parameters

## Reference outputs/accuracies: 

*Text Continuation* (`python run_llama.py --option generate`)
You should see continuations of the sentence `I have wanted to see this thriller for a while, and it didn't disappoint. Keanu Reeves, playing the hero John Wick, is...`. We will generate two continuations - one with temperature 0.0 (which should have a reasonably coherent, if unusual, completion) and one with temperature 1.0 (which is likely to be logically inconsistent and may contain some coherence or grammar errors).

*Zero Shot Prompting*
Zero-Shot Prompting for SST:

`python run_llama.py --option prompt --batch_size 10  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt [--use_gpu]`

Prompting for SST:
Dev Accuracy: 0.213 (0.000)
Test Accuracy: 0.224 (0.000)

Zero-Shot Prompting for CFIMDB:

`python run_llama.py --option prompt --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt [--use_gpu]`

Prompting for CFIMDB:
Dev Accuracy: 0.498 (0.000)
Test Accuracy: -

*Classification Finetuning*

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80  --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt [--use_gpu]`

Finetuning for SST:
Dev Accuracy: 0.414 (0.014)
Test Accuracy: 0.418 (0.017)

`python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10  --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt [--use_gpu]`

Finetuning for CFIMDB:
Dev Accuracy: 0.800 (0.115)
Test Accuracy: -

### Acknowledgement
This code is based on llama2.c by Andrej Karpathy. Parts of the code are also from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).
