## Index
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

<img src="https://user-images.githubusercontent.com/12748752/156908533-291c1992-92ad-440f-b715-78d5608e01d0.png" width=50% />

## BERT
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
* BERT is specificly at the Encoder side.
* Encoder->Nx term is going to be increased
* First BERT trained on Wiki dataset 70GB/80GB

* Language Modeling
   * Topic Modeling= Unidirectional Task . L->R
*  Language Model can be  Uni or Bidirectional depends on the masking staratigy
* Ram is playing Football[M]

> Masking can be anywhere in the sentence.
* Unidirectional and Bidirectional Masking

> #### Masking : Remove some words / token in the original corpus.
* The goal is to learn the context
* Not using any decoder in BERT

* BERT is a language model.
* EOS -> SEP
* SOS->cls
* 
---
1. Data loading
2. Data preprocessing-
   *  Masking Stratigy 
3. Tokenization
### Masking Stratigy
* My name is Arijit
* after masking
* My name is [Masking]
* 3 types of Masking Stratigy
   1. 80% of data covered with mask token
   2. 10% of data replaced with some random words.
   3. 10% of words will be kept unchanged.
*  Where Masking has to be applied
*  Randomly done
*  Which direction
   1.  L -> R
   2.  R -> L
   3.  R -> L and L -> R ( Bidirectional )  BERT

* Mask can be one word or one sentence.
---
---



[1]: http://example.com/ "Title"
[2]: http://example.org/ "Title"




