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

![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Masking Operation(removed words / token within the corpus)
* mask token works both ways (bidirectional)
* 1st BERT was on Wikipedia dataset(70GB or 80 GB)
* LSTM works well on small MBs datasets but GBs like BERT
### What does it can do?
* a Language model-has been tested 70+ language and worked well.
### EOS and SOS
In BERT we will refer the above as following
* EOS-SEP
* SOS-CLS(classification token)
## Tokenization Strategy
1) data loading
2) data processing 
3) Tokenization
* **80%** masked token
* **10%** chance of being **replaced** by a random words
* **10%** chance of being left alone/Unchanged (to **bias** the model toward the correct answer)
* Stopwords not need to be removed
## Where does the Masking hasbeen applied
* It is done by randomly
* direction L->R,R->L and L->R & R->L(BERT)

#### Word tokenization is more common than Sentence Tokenization


---
EOS and SOS
---
Different types of tasks by NLP
## 1. What is BERT used for?
BERT can be used on a wide variety of language tasks:

* Can determine how positive or negative a movie’s reviews are. (Sentiment Analysis)
* Helps chatbots answer your questions. (Question answering)
* Predicts your text when writing an email (Gmail). (Text prediction)
* Can write an article about any topic with just a few sentence inputs. (Text generation)
* Can quickly summarize long legal contracts. (Summarization)
* Can differentiate words that have multiple meanings (like ‘bank’) based on the surrounding text. (Polysemy resolution)

#### There are many more language/NLP tasks + more detail behind each of these.
> Fun Fact: You interact with NLP (and likely BERT) almost every single day!

NLP is behind Google Translate, voice assistants (Alexa, Siri, etc.), chatbots, Google searches, voice-operated GPS, and more.

## What is a Masked Language Model?
MLM enables/enforces bidirectional learning from text by masking (hiding) a word in a sentence and forcing BERT to bidirectionally use the words on either side of the covered word to predict the masked word. This had never been done before!

Fun Fact: We naturally do this as humans!

Masked Language Model Example:

Imagine your friend calls you while camping in Glacier National Park and their service begins to cut out. The last thing you hear before the call drops is:

Friend: “Dang! I’m out fishing and a huge trout just [blank] my line!”

Can you guess what your friend said??

You’re naturally able to predict the missing word by considering the words bidirectionally before and after the missing word as context clues (in addition to your historical knowledge of how fishing works). Did you guess that your friend said, ‘broke’? That’s what we predicted as well but even we humans are error-prone to some of these methods.

Note: This is why you’ll often see a “Human Performance” comparison to a language model’s performance scores. And yes, newer models like BERT can be more accurate than humans! 🤯

The bidirectional methodology you did to fill in the [blank] word above is similar to how BERT attains state-of-the-art accuracy. A random 15% of tokenized words are hidden during training and BERT’s job is to correctly predict the hidden words. Thus, directly teaching the model about the English language (and the words we use). Isn’t that neat?

Play around with BERT’s masking predictions:

## What is Next Sentence Prediction?
NSP (Next Sentence Prediction) is used to help BERT learn about relationships between sentences by predicting if a given sentence follows the previous sentence or not.

Next Sentence Prediction Example:

* Paul went shopping. He bought a new shirt. (correct sentence pair)
* Ramona made coffee. Vanilla ice cream cones for sale. (incorrect sentence pair)

In training, 50% correct sentence pairs are mixed in with 50% random sentence pairs to help BERT increase next sentence prediction accuracy.

Fun Fact: BERT is trained on both MLM (50%) and NSP (50%) at the same time.

##  BERT's performance on common language tasks
BERT has successfully achieved state-of-the-art accuracy on 11 common NLP tasks, outperforming previous top NLP models, and is the first to outperform humans! But, how are these achievements measured?

> NLP Evaluation Methods:
### SQuAD v1.1 & v2.0
SQuAD (**Stanford Question Answering Dataset**) is a reading comprehension dataset of around 108k questions that can be answered via a corresponding paragraph of Wikipedia text. BERT’s performance on this evaluation method was a big achievement beating previous state-of-the-art models and human-level performance:

## BERT - Tokenization and Encoding
### The [CLS],[SEP] and [PAD] Tokens
* Every word having unique token id that not be prediced but the tokens of the flllowing are always fixed
   * [CLS]-101 
   * [SEP]-102
   * [PAD]-0
---
## GPT is auto regressiove
---
