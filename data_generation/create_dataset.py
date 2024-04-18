import os
import configparser
import json
import pickle
import pandas as pd
import openai
import torch

import re
import unicodedata
import datetime

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser


current_date = datetime.datetime.now().date()
target_date = datetime.date(2024, 6, 12)
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"

openai.api_key = os.environ['OPENAI_API_KEY']

from definitions import *

from sklearn.model_selection import train_test_split
from collections import Counter

config = configparser.ConfigParser()
config.read(os.path.join(get_config_path()))

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


def collate_dataset(recipes, dialogues):
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


def create_prompt_schemas():
    prompt_preface = """
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
    """

    output_template = """
      You are given a recipe delimited by triple backticks.
      Following that is a question about the recipe and an answer is provided after the question.
      Return the following information :

      improved_question: If the question is very simple, you need to reframe the question and improve it by converting it to a question that requires more cooking related reasoning and details.
      improved_answer: Generate a detailed answer to the improved question in less than 100 words.
      follow_up: Array of 5 to 10 follow up questions and corresponding answers related to the the context and previous question and answer.

      text: {text}

      {format_instructions}
    """

    improved_question_schema = ResponseSchema(name="improved_question",
                                              description="If the question is very simple, you need to reframe the question and improve it by converting it to a question that requires more cooking related reasoning and details.")
    improved_answer_schema = ResponseSchema(name="improved_answer",
                                            description="Generate a detailed answer to the improved question in less than 150 words.")
    follow_up_schema = ResponseSchema(name="follow_up",
                                      description="follow_up: Array of 5 to 10 follow up questions and corresponding answers related to the the context and previous question and answer. format : [{'question': string, 'answer': string}]",
                                      type="array(objects)")

    response_schemas = [improved_question_schema,
                        improved_answer_schema,
                        follow_up_schema]

    return response_schemas, prompt_preface, output_template


def initiate_gpt_client():
    return openai.OpenAI(api_key=config['openai']['key'])


def get_completion(prompt, model=llm_model):
    client = openai.OpenAI(api_key=openai.api_key)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    response = completion.choices[0].message.content
    return response


def create_prompt(prompt_preface, recipe, question, answer):
  prompt = prompt_preface + f"\nRecipe : {recipe}\nQuestion : {question}\nAnswer : {answer}"
  return prompt


def create_qa_dataset(data):
  res = {}
  chat = ChatOpenAI(temperature=0.0, model=llm_model)
  response_schemas, prompt_preface, output_template = create_prompt_schemas()
  output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
  format_instructions = output_parser.get_format_instructions()

  for i, (title, v) in enumerate(data.items()):
    if i > 0 and i % 10 == 0:
      print(f"{i+1} recipes data generated...")
    instructions = v["instructions"]
    recipe = title + "\n" + instructions
    qna = v["qna"]
    res_qna = []
    for qa in qna:
      question, answer = qa["question"], qa["answer"]
      prompt_template = create_prompt(recipe, question, answer)
      prompt = ChatPromptTemplate.from_template(template=output_template)
      messages = prompt.format_messages(text=prompt_template,
                                format_instructions=format_instructions)
      response = chat(messages)
      output_dict = output_parser.parse(response.content)
      res_qna.append(output_dict)
    res[title] = {"instructions": instructions, "prompt_output": output_dict}
    with open(os.path.join(config['data']['data_path'], "generated_qa_dataset.pickle"), "wb") as f:
        pickle.dump(res, f)
  return res


def generate():
    recipes = load_recipes_dataset()
    dialogues = load_dialogue_dataset()
    recipe_dialogue_data = collate_dataset(recipes, dialogues)
    generated_dataset = create_qa_dataset(recipe_dialogue_data)
    return generated_dataset


def to_raw(string):
    return fr"{string}"


def convert_dataset_to_pandas(dataset):
    num_errors = 0
    dataset_to_convert = {'question': [], 'recipe_tag': [], 'recipe': [], 'student_ground_truth_answer': [],
                          'teacher_answer': []}
    recipes = load_recipes_dataset()
    for i, (recipe_title, item) in enumerate(dataset.items()):
        recipe_title = recipe_title.strip()
    instructions = item['instructions']
    student_qna = item['qna']
    teacher_qna = item['improved_qna']
    for j, (student_qna_item, teacher_qna_item) in enumerate(zip(student_qna, teacher_qna)):
        try:
            teacher_qna_item_dict = json.loads(to_raw(teacher_qna_item), strict=False)
            dataset_to_convert['recipe_tag'].append(i)
            dataset_to_convert['recipe'].append(recipe_title + "\n" + instructions)
            dataset_to_convert['question'].append(student_qna_item['question'])
            dataset_to_convert['student_ground_truth_answer'].append(student_qna_item['answer'])
            dataset_to_convert['teacher_answer'].append(teacher_qna_item_dict['Answer'])
        except Exception as e:
            num_errors += 1
            print("Exception {} at {} : {}".format(e, j, teacher_qna_item))
    print("Num error conversions : {}".format(num_errors))

    data_df = pd.DataFrame.from_dict(dataset_to_convert)
    assert len(data_df['question']) == len(data_df['recipe']) == len(data_df['student_ground_truth_answer']) == len(
        data_df['teacher_answer'])
    print("Length of converted dataset : {}".format(len(data_df['question'])))

    return data_df


def clean_words(sentence):
    sentence = str(sentence).lower()
    sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore').decode('utf-8',
                                                                                        'ignore')  # for converting é to e and other accented chars
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = re.sub(r"there's", "there is", sentence)
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    sentence = re.sub(r"'til", "until", sentence)
    sentence = re.sub(r"\"", "", sentence)
    sentence = re.sub(r"\'", "", sentence)
    sentence = re.sub(r' s ', "", sentence)
    sentence = re.sub(r"&39", "", sentence)  # the inshorts data has this in it
    sentence = re.sub(r"&34", "", sentence)  # the inshorts data has this in it
    sentence = re.sub(r"[\[\]\\0-9()\"$#%/@;:<>{}`+=~|.!?,-]", "", sentence)
    sentence = re.sub(r"&", "", sentence)
    sentence = re.sub(r"\\n", "", sentence)
    sentence = sentence.strip()
    return sentence


def preprocess(train_df, validation_df):
    print("Preprocessing data...\n")
    train_df["question"] = train_df["question"].apply(lambda x: clean_words(x))
    train_df["recipe"] = train_df["recipe"].apply(lambda x: clean_words(x))
    train_df["student_ground_truth_answer"] = train_df["student_ground_truth_answer"].apply(
        lambda x: clean_words(x))
    train_df["teacher_answer"] = train_df["teacher_answer"].apply(lambda x: clean_words(x))

    validation_df["question"] = validation_df["question"].apply(lambda x: clean_words(x))
    validation_df["recipe"] = validation_df["recipe"].apply(lambda x: clean_words(x))
    validation_df["student_ground_truth_answer"] = validation_df["student_ground_truth_answer"].apply(
        lambda x: clean_words(x))
    validation_df["teacher_answer"] = validation_df["teacher_answer"].apply(lambda x: clean_words(x))

    return train_df, validation_df


def prepare_dataset_for_distillation():
    data_path = config['data']['data_path']
    dir_name = os.path.join(data_path, "generated")
    if os.path.isdir(dir_name):
        if not os.listdir(dir_name):
            dataset = generate()
            data_df = convert_dataset_to_pandas(dataset)
            train_df, validation_df = train_test_split(data_df, test_size=0.2, random_state=0,
                                                       stratify=data_df[['recipe_tag']])
            print(len(Counter(data_df['recipe_tag'])))
            print(f"Number of training samples : {len(train_df)}\nNumber of validation samples : {len(validation_df)}")
            train_df, validation_df = preprocess(train_df, validation_df)
            train_df.to_pickle(
                os.path.join(data_path, "generated", "train_df.pickle"))
            validation_df.to_pickle(
                os.path.join(data_path, "preprocessed_data_files", "validation_df.pickle"))
        else:
            print("Fetching generated data...\n")
            train_df = pd.read_pickle(
                os.path.join(data_path, "generated", "train_df.pickle"))
            validation_df = pd.read_pickle(
                os.path.join(data_path, "generated", "validation_df.pickle"))
            print("Fetched generated data!")
        return train_df, validation_df
    else:
        print("Invalid path! Given directory doesn't exist")