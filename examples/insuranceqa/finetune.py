# Databricks notebook source
# MAGIC %md
# MAGIC ## Fine Tune Dolly
# MAGIC 
# MAGIC This fine-tunes the [Dolly](https://huggingface.co/databricks/dolly-v2-12b) model on
# MAGIC the Insurance QA dataset.
# MAGIC 
# MAGIC ```
# MAGIC   Licensed under the Apache License, Version 2.0 (the "License");
# MAGIC   you may not use this file except in compliance with the License.
# MAGIC   You may obtain a copy of the License at
# MAGIC 
# MAGIC       http://www.apache.org/licenses/LICENSE-2.0
# MAGIC 
# MAGIC   Unless required by applicable law or agreed to in writing, software
# MAGIC   distributed under the License is distributed on an "AS IS" BASIS,
# MAGIC   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# MAGIC   See the License for the specific language governing permissions and
# MAGIC   limitations under the License.
# MAGIC ```
# MAGIC 
# MAGIC Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B),
# MAGIC the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca).

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_id_or_path = "databricks/dolly-v2-3b"

model = AutoModelForCausalLM.from_pretrained(model_id_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# COMMAND ----------

!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb -O /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-3_11.5.1.109-1_amd64.deb -O /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb -O /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-3_10.2.4.109-1_amd64.deb -O /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb && \
  dpkg -i /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
  dpkg -i /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
  dpkg -i /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
  dpkg -i /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb

# COMMAND ----------

!pip install deepspeed datasets accelerate -q

# COMMAND ----------

tensorboard_display_dir = f"/tmp/runs"

%load_ext tensorboard
%tensorboard --logdir '{tensorboard_display_dir}'

# COMMAND ----------

torch.cuda.empty_cache()

!deepspeed \
    --num_gpus 8 \
    --module train \
    --input-model {model_id_or_path} \
    --deepspeed deepspeed.json \
    --epochs 10 \
    --local-output-dir /tmp \
    --dbfs-output-dir /tmp \
    --per-device-train-batch-size 20 \
    --per-device-eval-batch-size 20 \
    --logging-steps 10 \
    --save-steps 200 \
    --save-total-limit 20 \
    --eval-steps 100 \
    --warmup-steps 100 \
    --test-size 200 \
    --lr 1e-5 \
    --dataset-path "/tmp/data/insuranceqa"

# COMMAND ----------

!ls /tmp | grep checkpoint

# COMMAND ----------

!ls /tmp/checkpoint-800

# COMMAND ----------

dbutils.fs.cp("file:///tmp/checkpoint-800", "dbfs:/dolly-insuranceqa/", recurse = True)

# COMMAND ----------

!pip install accelerate

# COMMAND ----------

tokenizer = AutoTokenizer.from_pretrained("/tmp/checkpoint-800", padding_side = "left")
model = AutoModelForCausalLM.from_pretrained("/tmp/checkpoint-800", torch_dtype=torch.bfloat16).to("cuda")

# COMMAND ----------

import torch
from instruct_pipeline import InstructionTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

generate_text = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer, device = torch.device("cuda:0"))

res = generate_text("tell me all about health insurance.")
print(res[0]["generated_text"])

# COMMAND ----------


