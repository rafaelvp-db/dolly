# Databricks notebook source
!wget https://github.com/chatopera/insuranceqa-corpus-zh/raw/release/corpus/pool/answers.json.gz -O /tmp/answers.json.gz

# COMMAND ----------

# DBTITLE 1,Answers
import gzip
import json
import pandas as pd

# using gzip.decompress(s) method
s = gzip.open("/tmp/answers.json.gz", mode = "rb")
data_dict = json.loads(s.read().decode("utf-8"))
answers_df = pd.DataFrame.from_dict(data_dict, orient = "index")
answers_df = answers_df.drop("zh", axis = 1)

answers_df = answers_df.reset_index()#.rename(columns = {"level_0": "answer_id"}).drop("index", axis = 1)
answers_df = answers_df.rename(columns = {"index": "answer_id"})
answers_df["answer_id"] = pd.to_numeric(answers_df["answer_id"])
answers_df[answers_df["answer_id"] == 20798]

# COMMAND ----------

# DBTITLE 1,Questions
!wget https://github.com/chatopera/insuranceqa-corpus-zh/raw/release/corpus/pool/train.json.gz -O /tmp/train.json.gz

# COMMAND ----------

s = gzip.open("/tmp/train.json.gz", mode = "rb")
data_dict = json.loads(s.read().decode("utf-8"))
train_df = pd.DataFrame.from_dict(data_dict, orient = "index")
train_df = train_df.drop("zh", axis = 1)
train_df

# COMMAND ----------

train_df = train_df.explode("answers").drop("negatives", axis=1)
train_df

# COMMAND ----------

train_df = train_df.rename(columns = {"answers": "answer_id"})
train_df["answer_id"] = pd.to_numeric(train_df["answer_id"])
train_df = train_df.reset_index().rename(columns = {"index": "question_id"})
train_df["question_id"] = pd.to_numeric(train_df["question_id"])

# COMMAND ----------

train_df[train_df.question_id == 12887]

# COMMAND ----------

merge_df = train_df.merge(answers_df, left_on = "answer_id", right_on = "answer_id", suffixes = ["_question", "_answer"])
merge_df[merge_df.question_id == 12887]

# COMMAND ----------

output_df = merge_df.drop(["question_id", "answer_id", "domain"], axis = 1).rename(
    columns = {"question_id": "instruction", "en_answer": "response", "en_question": "instruction"})
output_df["context"] = ""
output_df["instruction"] = output_df["instruction"].apply(lambda x: str(x).lower())
output_df["response"] = output_df["response"].apply(lambda x: str(x).lower())
output_json = output_df.to_json(orient = "records", lines = True)

with open("data/insuranceqa.jsonl", "w") as file:
    file.write(output_json)

# COMMAND ----------

!pip install -q datasets

# COMMAND ----------

from datasets import Dataset, DatasetDict



# COMMAND ----------

dataset = Dataset.from_pandas(output_df, preserve_index = False)

# COMMAND ----------

dataset.save_to_disk("data/insuranceqa")
