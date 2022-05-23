## Index:
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
![light](https://user-images.githubusercontent.com/12748752/141935760-406edb8f-cb9b-4e30-9b69-9153b52c28b4.png)

## Bahdanau Attention
![dark](https://user-images.githubusercontent.com/12748752/141935752-90492d2e-7904-4f9f-a5a1-c4e59ddc3a33.png)
<img src="https://user-images.githubusercontent.com/12748752/168195356-8a08298c-9157-4656-9464-0dd4f7d56145.png" width=60% />
For machine translation in encoder-decoder architecture based on two RNNs for sequence to sequence learning. Specifically, the RNN encoder transforms a **variable-length sequence** into a **fixed-shape context variable**, then the RNN decoder generates the output (target) sequence _token by token_ based on the generated tokens and the context variable. However, even though not all the input (source) tokens are useful for decoding a certain token, the same context variable that encodes the entire input sequence is still used at each decoding step.

In a separate but related challenge of handwriting generation for a given text sequence, Graves designed a differentiable attention model to align text characters with the much longer pen trace, where the alignment moves only in one direction [Graves, 2013]. Inspired by the idea of learning to align, Bahdanau et al. proposed a differentiable attention model without the severe unidirectional alignment limitation [Bahdanau et al., 2014]. When predicting a token, if not all the input tokens are relevant, the model aligns (or attends) only to parts of the input sequence that are relevant to the current prediction. This is achieved by treating the context variable as an output of attention pooling.
