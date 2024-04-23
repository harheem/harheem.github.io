---
layout: page
title: "Prompt Tuning"
subtitle: "The Power of Scale for Parameter-Efficient Prompt Tuning"
date:   2024-02-19 20:13:50 +0530
categories: LLM
author: "Harheem Kim"
---

# Prompt Tuning

> **PEFT(Parameter-Efficient Fine-Tuning)는 적은 수의 파라미터를 학습하는것만으로 모델 전체를 파인튜닝하는 것과 유사한 효과를 누릴 수 있도록 해줍니다. PEFT 방법 중 하나인 Prompt Tuning에 대해서 알아봅시다.**
> 

[https://arxiv.org/pdf/2104.08691.pdf](https://arxiv.org/pdf/2104.08691.pdf)

## **프롬프트 튜닝이란?**

언어 모델을 특정 작업에 맞게 조정하기 위해 사용되는 기술입니다. 기존의 방식은 모델을 특정 작업에 맞게 전체적으로 조정해야 했지만, 프롬프트 튜닝은 모델의 핵심 부분을 그대로 유지하면서 작업 특화 부분만 조정합니다. 이는 모델의 '냉동'(frozen) 상태를 유지하면서도 필요한 부분에만 초점을 맞추어 효율성을 높이는 방법입니다.

![*Prompt tuning retains the strong task performance of model tuning, while keeping the pre-trained model frozen, enabling efficient multitask serving.*](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F2a330106-7d16-49d5-9057-343dfb0cb92c%2F06fcee26-22da-429a-9e1b-46c98535ed1d%2FUntitled.png?table=block&id=1be7483c-133f-42ec-b874-da2bdfce41bb&spaceId=2a330106-7d16-49d5-9057-343dfb0cb92c&width=2000&userId=93a922d0-0a24-445b-bddf-8a085b93d655&cache=v2)

*Prompt tuning retains the strong task performance of model tuning, while keeping the pre-trained model frozen, enabling efficient multitask serving.*

## **작동 원리**

소프트 프롬프트는 학습 가능한 ‘벡터’로 이루어져 있습니다. 이 벡터들은 입력 텍스트와 결합되어 모델의 입력으로 사용됩니다. 이 벡터들은 기존 어휘에 속하지 않는 '가상의 토큰(virtual tokens)'으로서 작동하며, 모델의 기존 파라미터를 변경하지 않고도 특정 작업에 대한 모델의 반응을 조정할 수 있습니다. 모델은 이 입력을 기반으로 예측을 수행하고, 이 과정에서 오차를 계산하여 소프트 프롬프트를 최적화합니다. 이 방법을 통해, 다양한 작업에 대한 지식을 효과적으로 흡수하고 적용할 수 있게 됩니다.

![Untitled](https://www.notion.so/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F2a330106-7d16-49d5-9057-343dfb0cb92c%2F44052b28-0850-4572-a0dc-2d28dbaa6dcd%2FUntitled.png?table=block&id=8de0b385-4ab8-4411-a713-0b03e716c3a0&spaceId=2a330106-7d16-49d5-9057-343dfb0cb92c&width=2000&userId=93a922d0-0a24-445b-bddf-8a085b93d655&cache=v2)

먼저, 소프트 프롬프트를 고정 길이(e.g., 20 tokens long)의 벡터 시퀀스로 초기화합니다. 이 벡터들은 모델의 입력 텍스트 앞에 배치됩니다.

모델이 입력을 처리할 때, 이 소프트 프롬프트 벡터들도 함께 처리됩니다. 모델이 예측을 수행하면, 예측 결과와 실제 타겟 간의 오차를 계산하여 이 오차를 사용해 소프트 프롬프트 벡터를 업데이트합니다.

## 간단한 코드

---

```python
import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """appends learned embedding to 

        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                               n_tokens, 
                                                                               random_range, 
                                                                               initialize_from_vocab))
            
    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """initializes learned embedding

        Args:
            same as __init__

        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)
            
    def forward(self, tokens):
        """run forward pass

        Args:
            tokens (torch.long): input tokens before encoding

        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """

				# Changes: Apply word embeddings to the entire set of input tokens without slicing
        input_embedding = self.wte(tokens)
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)
```

출처: [https://github.com/kipgparker/soft-prompt-tuning](https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py)

이 코드는 PyTorch를 사용하여 'SoftEmbedding'이라는 신경망 모듈을 정의합니다. 

이 모듈의 주요 목적은 기존 트랜스포머 모델의 워드 임베딩(word embedding)에 추가적인 학습 가능한 임베딩을 결합하는 것입니다. 이를 통해 특정 작업에 대한 모델의 성능을 향상시킬 수 있습니다. 

출처 코드에서는 `input_embedding = self.wte(tokens[:, self.n_tokens:])` 로 소프트 프롬프트 토큰의 길이만큼 원본 임베딩을 잘라서 결합하였지만, 저는 원본 임베딩에 추가된 임베딩을 결합해서 사용하기 위해  다음과 같이 코드를 변경하였습니다: `input_embedding = self.wte(tokens)`

코드를 자세히 살펴보겠습니다.

`**SoftEmbedding**`

이 클래스는 `nn.Module`을 상속받아 PyTorch의 신경망 모듈로 정의됩니다.

`**__init__**`

- `wte (nn.Embedding)`: 기존 트랜스포머 모델의 워드 임베딩을 나타냅니다.
- `n_tokens (int)`: 학습 가능한 추가 토큰의 수입니다. 이 값이 10일 때, 10개의 추가 임베딩 토큰이 생성됩니다.
- `random_range (float)`: 임베딩을 초기화할 때 사용되는 범위입니다. 이 값이 0.5일 때, 각 임베딩 값은 -0.5 ~ 0.5 사이의 범위에서 무작위로 초기화됩니다.
- `initialize_from_vocab (bool)`: 기존 어휘에서 임베딩을 초기화할지 여부를 결정합니다. 이 값은 아래 `initialize_embedding` 에서 어떻게 사용되는지 알 수 있습니다.
- `learned_embedding`: 특정 작업에 특화된 정보를 포함할 수 있도록 설계된 새로운 임베딩입니다. 추가적인 학습 가능한 임베딩을 정의하며, 초기화 방법은 `initialize_embedding` 메서드에 의해 결정됩니다.

 **`initialize_embedding`**

- 이 메서드는 추가 임베딩을 초기화하는 데 사용됩니다.
- `initialize_from_vocab`가 `True`이면 기존의 워드 임베딩(wte)에서 처음 `n_tokens`만큼을 복사하여 사용합니다. 이 방법은 기존 어휘에 기반한 임베딩을 사용하기 때문에, 모델이 이미 학습한 언어적 특성을 유지하도록 합니다.
- `False`인 경우, 지정된 `random_range`를 사용하여 임베딩을 무작위로 초기화합니다. 이 방법은 모델이 이전에 보지 못한 새로운 종류의 데이터나 작업에 대응해야 할 때 유용합니다.

`**forward**`

- 모델이 입력 데이터를 어떻게 처리하는지 정의합니다. 이 메서드는 입력 토큰을 받아 추가적인 학습된 임베딩과 함께 원래의 워드 임베딩을 결합합니다.
- `tokens`: 입력 데이터를 나타냅니다. 이는 모델이 처리할 원시 텍스트를 토큰화한 것입니다.
- `learned_embedding`은 모든 입력에 대해 반복되며, 기존 입력 임베딩과 연결됩니다.
- 최종적으로, 학습된 임베딩과 입력 임베딩이 연결되어 반환됩니다.

## peft, transformers 라이브러리를 활용한 예시

---

시작하기 전에 **`peft`**, **`transformers`**, **`datasets`, `torch`** 등 필요한 라이브러리를 설치합니다.

```python
!pip install -q peft transformers datasets torch
```

사용할 모델과 토크나이저를 정의합니다. 이 예시에서는 **`bigscience/bloomz-560m`**을 모델과 토크나이저로 사용하였습니다. **`PromptTuningConfig`**를 정의하여 작업 유형, 가상 토큰의 수, 초기화 텍스트, 토크나이저 이름 또는 경로 등의 세부 정보를 지정합니다.

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

device = "cuda"
model_name_or_path = "bigscience/bloomz-560m"
tokenizer_name_or_path = "bigscience/bloomz-560m"
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path=model_name_or_path,
)

dataset_name = "twitter_complaints"
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)
text_column = "Tweet text"
label_column = "text_label"
max_length = 64
lr = 3e-2
num_epochs = 50
batch_size = 8
```

이 예제에서는 **`ought/raft`**의 **`twitter_complaints`**라는 데이터셋을 사용합니다. 이 데이터셋은 트위터의 트윗들을 포함하고 있으며, 감정 분석이나 텍스트 분류를 위한 연구에 주로 사용됩니다. 데이터셋을 전처리하는 코드는 생략합니다. 자세한 내용은 참고 코드를 확인해주세요.

```python
from datasets import load_dataset

dataset = load_dataset("ought/raft", dataset_name)

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
print(classes)
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
print(dataset)
dataset["train"][0]
```

모델을 초기화합니다. `print_trainable_parameters()`로 훈련 가능한 파라미터들을 확인할 수 있습니다.

```python
# creating model
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model
```

모델의 파라미터를 최적화하기 위해 AdamW 옵티마이저를 사용합니다. 학습률(lr)로 학습 과정에서 얼마나 큰 단계로 가중치를 업데이트할지 결정할 수 있습니다. 

```python
# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
```

전체 데이터셋에 대해 학습을 수행합니다. 훈련된 모델의 성능을 확인하기 위해 loss와 perplexity를 확인합니다. 

```python
# training and evaluation
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        #         print(batch)
        #         print(batch["input_ids"].shape)
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")
```

참고 코드: [https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_prompt_tuning_clm.ipynb](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_prompt_tuning_clm.ipynb)

문서: [https://huggingface.co/docs/peft/v0.8.2/en/package_reference/prompt_tuning#peft.PromptTuningConfig](https://huggingface.co/docs/peft/v0.8.2/en/package_reference/prompt_tuning#peft.PromptTuningConfig)

## 마무리

---

프롬프트 튜닝의 가장 큰 장점은 효율성입니다. 전체 모델을 다시 학습하지 않고도, 매우 적은 양의 파라미터만으로도 특정 작업에 대해 높은 성능을 낼 수 있습니다. 또한, 다양한 작업에 하나의 모델을 사용하여 리소스 활용도가 높아집니다. 특히 대규모 모델에서 이 방법의 효과가 크게 나타납니다.

구글 연구팀의 블로그에 따르면, 프롬프트 튜닝을 적용한 모델은 특정 도메인의 데이터로 학습한 후, 관련된 다른 도메인의 작업에 대해 '제로-샷' 평가를 수행했을 때 더 높은 정확도를 보였습니다. 예를 들어, 'Quora Question Pairs' 작업으로 학습된 모델이 'MRPC'(뉴스 기사의 문장이 서로 다른 방식으로 표현되었는지 판별하는 작업) 작업에서도 높은 성능을 보였습니다.

이러한 결과는 소프트 프롬프트 튜닝이 모델의 일반화 능력을 향상시키고, 특정 도메인에 과도하게 최적화되지 않도록 하는데 도움을 준다는 것을 시사합니다. 따라서, 언어 모델을 다양한 작업에 적용하고자 할 때 프롬프트 튜닝은 매우 유용한 도구가 될 수 있습니다.

더 자세한 정보와 연구 결과는 [Guiding Frozen Language Models with Learned Soft Prompts](https://blog.research.google/2022/02/guiding-frozen-language-models-with.html)를 참고하세요.

## Reference

[https://arxiv.org/pdf/2104.08691.pdf](https://arxiv.org/pdf/2104.08691.pdf)

[https://4n3mone.tistory.com/7](https://4n3mone.tistory.com/7)