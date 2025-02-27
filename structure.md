# Structure of Llama

## llama.py
This file contains the Llama2 model whose backbone is the [transformer](https://arxiv.org/pdf/1706.03762.pdf).

### Attention
The multi-head attention layer of the transformer. This layer maps a query and a set of key-value pairs to an output. The output is calculated as the weighted sum of the values, where the weight of each value is computed by a function that takes the query and the corresponding key.
$$Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$$

Llama2 uses a modified version of this procedure called [Grouped-Query Attention](https://arxiv.org/abs/2305.13245) where, instead of each attention head having its own "query", "key", and "vector" head, some groups of "query" heads share the same "key" and "vector" heads.

### LlamaLayer
This corresponds to one transformer layer which has 
1. layer normalization of the input (via Root Mean Square layer normalization)
2. self-attention on the layer-normalized input
3. a residual connection (i.e., add the input to the output of the self-attention)
4. layer normalization on the output of the self-attention
5. a feed-forward network on the layer-normalized output of the self-attention
6. a residual connection from the unnormalized self-attention output added to the
    output of the feed-forward network

### Llama
This is the Llama model that takes in input ids and returns next-token predictions and contextualized representation for each word. The structure of ```Llama``` is:
1. an embedding layer that consists of token embeddings ```tok_embeddings```.
2. llama encoder layer which is a stack of ```config.num_hidden_layers``` ```LlamaLayer```
3. a projection layer for each hidden state which predicts token IDs (for next-word prediction)
4. a "generate" function which uses temperature sampling with top-p sampling to generate long continuation strings.
The desired outputs are
1. ```logits```: logits (output scores) over the vocabulary, predicting the next possible token at each point
2. ```hidden_state```: the final hidden state at each token in the given document

## classifier.py
This file contains the pipeline to 
* load a pretrained model
* generate an example sentence (to verify that your implemention works)
* call the Llama2 model to encode the sentences for their contextualized representations
* feed in the encoded representations for the sentence classification task
* fine-tune the Llama2 model on the downstream tasks (e.g. sentence classification)

### LlamaSentClassifier
This class is used to
* encode the sentences using Llama2 to obtain the hidden representation from the final word of the sentence.
* classify the sentence by applying dropout to the pooled-output and project it using a linear layer.

## optimizer.py  (to be implemented)
This is where `AdamW` is defined.
You will need to update the `step()` function based on [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) and [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
There are a few slight variations on AdamW, pleae note the following:
- The reference uses the "efficient" method of computing the bias correction mentioned at the end of section 2 "Algorithm" in Kigma & Ba (2014) in place of the intermediate m hat and v hat method.
- The learning rate is incorporated into the weight decay update (unlike Loshchiloc & Hutter (2017)).
- There is no learning rate schedule.

## rope.py (to be implemented)
Rotary positional embeddings implemented here.

## base_llama.py
This is the base class for the Llama model.

## tokenizer.py
This is the tokenizer used.

## config.py
This is where the configuration class is defined.

## utils.py
This file contains utility functions for various purposes.
 
## Reference
[Vaswani el at. + 2017] Attention is all you need https://arxiv.org/abs/1706.03762
[Touvron el at. + 2023] Llama 2: Open Foundation and Fine-Tuned Chat Models https://arxiv.org/abs/2307.09288
