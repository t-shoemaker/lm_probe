About
=====

This code is for `lm_head`, a little tool for training linear probes on neural
language models. Use it to isolate model behavior via classification tasks.

**Features:**

+ Flexible probe configuration for classification tasks using logistic
  regression
+ Simultaneous, online training for multiple probes with minimal overhead for
  activation caches
+ Pre/post-processing for model activations (probes default to mean pooling)
+ Metrics tracking for probe performance

Probes are trained using [`scikit-learn`][scikit]'s `SGDClassifier`. The tool
relies on [`nnsight`][nnsight] to trace model activations. All you need is your
own text and labels.

**Caveats:**

+ This is a small tool for quickly standing up probe experiments; brittleness
  and sharp corners are likely
+ Probes are limited to classification tasks

[scikit]: https://scikit-learn.org/stable/
[nnsight]: https://nnsight.net/


Installation
------------

While a full set of dependencies is in `pyproject.toml`, I've made no attempt
to configure installation for different CPU/GPU setups; install this tool into
an environment that already has whatever PyTorch configuration you use.

To install with `pip`, run the following:

```sh
pip install git+https://github.com/t-shoemaker/lm_probe
```

Or, with `pixi`:

```sh
pixi add --pypi "lm_probe @ git+https://github.com/t-shoemaker/lm_probe"
```


Usage
-----

You can run your probes on any `LanguageModel` from `nnsight` (which wraps
Hugging Face models and PyTorch modules).

```py
import torch
import pandas as pd
from nnsight import LanguageModel
from lm_probe import ProbeConfig, ProbeRunner, ProbeDataset


gpt2 = LanguageModel("gpt2", device_map="auto")
```


**Data preparation**

Probes learn directly from activations, so you'll need to tokenize your texts
separately. Below, we use a sample of [IMDB reviews][imdb]. The probes will
attempt to classify positive and negative reviews using model activations. We
wrap the tokenized reviews in a `ProbeDataset`, which handles batching for the
model.

[imdb]: https://huggingface.co/datasets/scikit-learn/imdb

```py
# Download the data, map string labels to integers, and sample
df = pd.read_csv("hf://datasets/scikit-learn/imdb/IMDB Dataset.csv")
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
df = df.sample(1_000)

# Tokenize with the model
reviews, labels = df["review"].tolist(), df["label"].tolist()
inputs = gpt2.tokenizer(
    reviews, padding="max_length", truncation=True, return_tensors="pt"
)

# Wrap everything up in a ProbeDataset and do a train/test split
ds = ProbeDataset(
    inputs["input_ids"], inputs["attention_mask"], labels
)
train_set, test_set = ds.tt_split(test_size=0.1)
```


**Probe configuration**

With `nnsight`, you can extract activations from any part of a neural network.
This functionality is exposed in `lm_probe` with stringified submodule names.
If, for example, you wanted to extract output activations from layer 5's MLP,
you'd write `transformer.h.5.mlp.output`.

```py
gpt2
>> GPT2LMHeadModel(
>>   (transformer): GPT2Model(
>>     (wte): Embedding(50257, 768)
>>     (wpe): Embedding(1024, 768)
>>     (drop): Dropout(p=0.1, inplace=False)
>>     (h): ModuleList(
>>       (0-11): 12 x GPT2Block(
>>         (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
>>         (attn): GPT2SdpaAttention(
>>           (c_attn): Conv1D()
>>           (c_proj): Conv1D()
>>           (attn_dropout): Dropout(p=0.1, inplace=False)
>>           (resid_dropout): Dropout(p=0.1, inplace=False)
>>         )
>>         (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
>>         (mlp): GPT2MLP(
>>           (c_fc): Conv1D()
>>           (c_proj): Conv1D()
>>           (act): NewGELUActivation()
>>           (dropout): Dropout(p=0.1, inplace=False)
>>         )
>>       )
>>     )
>>     (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
>>   )
>>   (lm_head): Linear(in_features=768, out_features=50257, bias=False)
>>   (generator): WrapperModule()
>> )
```

Assign that submodule name to the first argument of a `ProbeConfig` along with
the unique set of classes you're looking for, and you're ready to perform a
probe.

```py
classes = set(labels)
submodule = "transformer.h.5.mlp.output"
config = ProbeConfig(submodule, classes)
```


**Training**

Use a `ProbeRunner` to train the probe. It will handle pre-/postprocessing for
activations as well as the control flow for batching, evaluation, early
stopping, etc.

```py
runner = ProbeRunner(gpt2, [config])
runner.fit_probes(train_set, test_set, batch_size=8)
>> 2024-12-04 09:47:11 | INFO | Early stopping after 18 steps for transformer.h.5.mlp.output
>> 2024-12-04 09:47:11 | INFO | Finished training 1 probe(s). Computing metrics
```

Inspect the results with the `.metrics` attribute. This holds a DataFrame of 
metrics, which record a probe's performance on the testing set.

```py
print(runner.metrics.to_string())
>>                                 loss  accuracy  precision    recall        f1  matthews
>> transformer.h.5.mlp.output  4.505457     0.875   0.881834  0.875941  0.874613  0.757752
```


**Using trained probes**

The runner's `.get_probe_features()` method extracts its probes' features for
some input and returns them in a simple dictionary.

```py
docs = test_set[0:5]
features = runner.get_probe_features(docs["input_ids"], docs["attention_mask"])
features[submodule].shape
>> (5, 768)
```

Index a `ProbeRunner` by a submodule to get the corresponding probe. With that,
you can make a prediction on features.

```py
probe = runner[submodule]
probe.predict(features[submodule])
>> array([0, 1, 0, 0, 0])
```


**Parallelized Probing**

The real point of `lm_probe` is that it parallelizes probe training. It does
this with minimal activation caching, relying instead on `nnsight` to trace
model layers during processing. While this front-loads memory costs during
model processing, it enables quick experimentation with different probe
configurations without having to keep all activations in memory. With that in
mind, we initialize a few probes below.

```py
configs = [
    ProbeConfig(
        "transformer.h.11.output",
        classes,
        test_size=0.5,
        patience=10,
        warmup_steps=25,
    ),
    ProbeConfig(
        "transformer.h.4.mlp.output",
        classes,
        test_size=0.5,
        patience=10,
        warmup_steps=25,
    ),
    ProbeConfig(
        "transformer.h.7.attn.output",
        classes,
        test_size=0.5,
        patience=10,
        warmup_steps=25,
    ),
]
```

Time to train.

```py
runner = ProbeRunner(gpt2, configs)
runner.fit_probes(train_set, test_set, batch_size=8)
>> 2024-12-04 09:50:59 | INFO | Early stopping after 38 steps for transformer.h.4.mlp.output
>> 2024-12-04 09:51:28 | INFO | Early stopping after 42 steps for transformer.h.7.attn.output
>> 2024-12-04 09:51:58 | INFO | Early stopping after 46 steps for transformer.h.11.output
>> 2024-12-04 09:51:58 | INFO | Finished training 3 probe(s). Computing metrics
```

Which set of model activations leads to the most accurate classification?

```py
print(
    runner.metrics
    .sort_values(["loss", "matthews"], ascending=[True, False])
    .to_string()
)
>>                                   loss  accuracy  precision    recall        f1  matthews
>> transformer.h.11.output       6.590683  0.815217   0.814490  0.815834  0.814774  0.630322
>> transformer.h.4.mlp.output    6.870814  0.807065   0.806108  0.807062  0.806435  0.613169
>> transformer.h.7.attn.output  11.722360  0.668478   0.673981  0.672990  0.668390  0.346969
```
