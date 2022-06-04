## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Bidirectional Encoder Representations from Transformers (BERT)
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
The [**BERT** paper](https://arxiv.org/abs/1810.04805) by Jacob Devlin and other Google researchers also demonstrates the effectiveness of **self-supervised pretraining** on a large corpus, using a similar architecture to GPT but **nonmasked Multi-Head Attention layers** (like in the **Transformer’s encoder**). 

This means that the _model is naturally bidirectional_; hence the **B** in **BERT (Bidirectional Encoder Representations from Transformers)**. Most importantly, the authors proposed **two pretraining** tasks that explain most of the model’s strength:

### <ins><i>1. Masked language model (MLM)</i></ins>
Each word in a sentence has a 15% probability of being masked, and the model is trained to predict the masked words. 
#### For example:
If the original sentence is -<img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\textbf{She&space;had&space;fun&space;at&space;the&space;birthday&space;party}}" title="https://latex.codecogs.com/svg.image?{\color{Purple} \textbf{She had fun at the birthday party}}" align="center"/>  
* Then the model may be given the sentence 
<img src="https://latex.codecogs.com/svg.image?{\color{Purple}&space;\mathbf{She&space;\left<&space;mask\right>&space;fun\&space;at\&space;the&space;\left<&space;mask\right>&space;party}}" title="" align="center"/>  and it must predict the words “**had**” and “**birthday**” (the other outputs will be ignored). 
* To be more precise, each selected word has an **80%** chance of being **masked**, a **10%** chance of being **replaced** by a random word (to reduce the discrepancy between pretraining and fine-tuning, since the model will not see **`<mask>`** tokens during fine-tuning), and a **10%** chance of being left alone (to **bias** the model toward the correct answer).


### <ins><i>2. Next sentence prediction (NSP)</i></ins>
* The model is trained to predict whether two sentences are consecutive or not. 
* For example, it should predict that “**The dog sleeps**” and “**It snores loudly**” are consecutive sentences, while “**The dog sleeps**” and “**The Earth orbits the Sun**” are not consecutive. 
* This is a challenging task, and it significantly improves the performance of the model when it is fine-tuned on tasks such as question answering or entailment.






We have introduced several word embedding models for natural language understanding. After pretraining, the output can be thought of as a matrix where each row is a vector that represents a word of a predefined vocabulary. In fact, these word embedding models are all **context-independent**. Let us begin by illustrating this property.

### From Context-Independent to Context-Sensitive
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

Recall the experiments in Section 14.4 and Section 14.7. For instance, word2vec and GloVe both assign the same pretrained vector to the same word regardless of the context of the word (if any). Formally, a context-independent representation of any token  is a function  that only takes  as its input. Given the abundance of polysemy and complex semantics in natural languages, context-independent representations have obvious limitations. For instance, the word “crane” in contexts “a crane is flying” and “a crane driver came” has completely different meanings; thus, the same word may be assigned different representations depending on contexts.

This motivates the development of context-sensitive word representations, where representations of words depend on their contexts. Hence, a context-sensitive representation of token  is a function  depending on both  and its context . Popular context-sensitive representations include TagLM (language-model-augmented sequence tagger) [Peters et al., 2017b], CoVe (Context Vectors) [McCann et al., 2017], and ELMo (Embeddings from Language Models) [Peters et al., 2018].

For example, by taking the entire sequence as the input, ELMo is a function that assigns a representation to each word from the input sequence. Specifically, ELMo combines all the intermediate layer representations from pretrained bidirectional LSTM as the output representation. Then the ELMo representation will be added to a downstream task’s existing supervised model as additional features, such as by concatenating ELMo representation and the original representation (e.g., GloVe) of tokens in the existing model. On one hand, all the weights in the pretrained bidirectional LSTM model are frozen after ELMo representations are added. On the other hand, the existing supervised model is specifically customized for a given task. Leveraging different best models for different tasks at that time, adding ELMo improved the state of the art across six natural language processing tasks: sentiment analysis, natural language inference, semantic role labeling, coreference resolution, named entity recognition, and question answering.
