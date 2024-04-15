import pickle
import json
import os
import re
import time
import pandas as pd
from os import listdir
from os.path import isfile, join

import torch
from definitions import *

import configparser
config = configparser.ConfigParser()
config.read(os.path.join(get_config_path()))

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from transformers import pipeline, AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
import time
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel, PeftConfig, LoraConfig, TaskType

import pytorch_lightning as pl
print(f"pytorch lightning version : {pl.__version__}")

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_recipes_dataset():
    with open(os.path.join(config['data']['data_path'], "recipes.pickle"), "rb") as f:
        recipes_dataset = pickle.load(f)

    formated_recipes_dataset = {}
    for k, v in recipes_dataset.items():
        formated_recipes_dataset[k.strip()] = v

    return formated_recipes_dataset


def load_dialogue_dataset():
    with open(os.path.join(config['data']['data_path'], "cleaned_wizard_of_tasks_cooking.json"), "rb") as f:
        dialogue_dataset = json.load(f)
    return dialogue_dataset


def collate_dataset():
    recipes = load_recipes_dataset()
    dialogues = load_dialogue_dataset()

    recipe_dialogue_data = {}
    for i, (k, v) in enumerate(dialogues.items()):
        title = " ".join(v['document_url'].split("/")[-1].split("-")).strip()
        if title in recipes:
            instructions = dialogues[title]
            qna = []

            turn = 0
            while turn + 1 < len(v['turns']):
                question, answer = None, None
                student_turn = v['turns'][turn]
                teacher_turn = v['turns'][turn + 1]

                if student_turn["role"] == "student":
                    question = student_turn["text"]
                if teacher_turn["role"] == "teacher":
                    answer = teacher_turn["text"]
                qna.append({"question": question, "answer": answer})
                turn += 2

            recipe_dialogue_data[title] = {"instructions": instructions, "qna": qna}
    return recipe_dialogue_data


def construct_prompt(recipe, question, answer):
    prompt = """
    You are given a recipe delimited by triple backticks.
    Following that is a question about the recipe and an answer is provided after the question.
    If the question is very simple, you need to reframe the question and improve it by converting it to a question that requires more cooking related reasoning and details.
    After that generate a detailed answer to the improved question in less than 300 words. Format the output in valid json format.

    Example :
      Sample Input:
        ```Recipe :labneh fresh herbs and olive oil',
        Instructions : Line a strainer with a double layer of cheesecloth and suspend over a bowl.
        Spoon in yogurt. Refrigerate and let drain for at least 2 hours. Discard liquid.
        The longer the yogurt drains, the thicker the cheese will be. For a thicker spread, drain covered yogurt overnight in the refrigerator.
        Transfer to a bowl. Add oil, tarragon, basil, chives, thyme, zest, salt and pepper, and whisk until blended.
        Let sit for 15 minutes to allow the flavors to meld.
        Taste and adjust seasoning with salt and pepper.
        Labneh will keep in an airtight container in the refrigerator for up to 5 days.```

        Question : Can I let the ingredients sit for longer to make the flavors stronger?
        Answer : Only 15 minutes is needed for the flavors to meld.

      Sample Output:
        {
          "Question": "How does the duration of resting the labneh with herbs and olive oil influence the development of flavors, and what are the potential effects of letting it sit for longer than the recommended 15 minutes?",
          "Answer": "The resting period after blending the labneh with herbs and olive oil is critical for allowing the flavors to meld. The process involves the diffusion of essential oils from the herbs into the labneh, enhancing its flavor. Allowing the mixture to sit for the recommended 15 minutes usually suffices for the flavors to combine harmoniously. Extending this period can lead to a more pronounced flavor profile, as the herbs continue to release their oils, deepening the overall taste. However, letting it sit for too long could potentially overpower the delicate balance of the labneh, with dominant flavors from herbs like tarragon or thyme possibly becoming too intense. Moreover, the texture of the labneh might be affected if the acidic components of the herbs begin to further break down the dairy proteins, potentially altering its creamy consistency.",
          "Critique": "The answer provides a detailed explanation of the effects of resting time on flavor development in labneh. However, it lacks specific guidelines on how much longer one can let it sit before the flavors become overpowering or the texture changes.",
          "Followup": {
          {
            "Question": What are the signs that the labneh has been overflavored by the herbs?",
            "Answer": "Signs that the labneh has been overflavored include a sharp or bitter taste, particularly from herbs like thyme and tarragon. The freshness and creaminess of the labneh may be overshadowed by an overwhelming herbal presence."
          },
          {
            "Question": "How might the texture of labneh change with extended resting times, and why?",
            "Answer": "With extended resting times, the texture of the labneh could become slightly grainy or watery. This occurs because the longer exposure to herbs, which contain acids and other compounds, can further break down the protein structure of the yogurt, affecting its smoothness and consistency."
          }
        }
    """
    prompt = prompt + f"Recipe : {recipe}\nQuestion : {question}\nAnswer : {answer}"
    return prompt







