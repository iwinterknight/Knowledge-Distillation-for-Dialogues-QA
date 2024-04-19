# Knowledge-Distillation-for-Dialogues-QA

This repo contains :
1. Code to build a specialist LLM for Cooking tasks, fine-tuned with knowledge distillation from an open-source generalist LLM's (GPT 3.5 Turbo) predictions.
2. Code to augment Amazon's Wizard of Tasks[^1] dataset by generating more elaborate Question Answer turns for the recipes.
3. Retrieval Augmented Generation(RAG) pipelines using `Parent Document Retriever` and `Contextual Compression Retriever` and TruLens evaluation for the 3 RAG Triad evaluations : context relevance, groundedness and answer relevance.



**Knowledge Distillation**

Instruction fine-tuned LLMs are capable of in-context learning and generalize well to tasks they have not encountered during the fine-tuning stage. However this capacity to generalize is seen when the order of parameters is large.

<p align="center">
 <img width="789" alt="In Context Learning" src=https://github.com/iwinterknight/Knowledge-Distillation-for-Dialogues-QA/assets/37212007/a501eb92-a098-4516-8b21-6545e4ae2bf3>
 <center> Effect of number of model parameters on in-context learning. </center>
</p>

Image Source : [here](https://arxiv.org/abs/2206.07682)


Knowledge distillation is an important means to train smaller models(student models) to perform well on domains specific tasks, by aligning their learning with specialist models or open sourced large generalist LLMs(teacher models). This is done by jointly training the student model on the outputs of teacher model and the ground truth. The teacher model loss is essentially a Kullback Leibler divergence loss which guides the student model's output distribution to align with the teacher model's output distribution. We set a temperature value for the student-teacher loss so as to allow some discrepancy between their outputs.

<p align="center">
 <img width="789" alt="Knowledge Distillation" src=https://github.com/iwinterknight/Knowledge-Distillation-for-Dialogues-QA/assets/37212007/815e64d4-bbac-4495-b614-f15e3503f7ab> 
 <center> Knowledge Distillation </center>
</p>


**Dataset Generation**

LangChain is leveraged to prompt gpt3.5-turbo API to augment the Amazon Wizard of Tasks[^1] Cooking dataset with more detailed question answers about the task. These questions typically capture 2 aspects of task related commonsense reasoning, physical (eg. kitchen tool, ingredient substitutes etc.) and temporal (eg. duration, ordering of steps, frequency of steps).

Here are a few examples of the type of questions generated through prompting :
```
1. Physical Commonsense Reasoning Questions :
Kitchen Tool Physical Commonsense Question: "I do not have a scraper or a chopper. How do I scrape the bowl?"
Answer: "You could use any other sharp object like a kitchen knife or even a fork"

Kitchen Tool Physical Commonsense Question: "I have a small vessel, can I boil the water in that?"
Answer: "You will need a somewhat large vessel to prevent the water from spilling over while boiling"

Ingredient Physical Commonsense Question: "I don't have an extra box and the packet of oats did not come in a ziplock bag. How should I store the left-over oats?"
Answer: "You could try tying the cut portion of the packet or sealing the opening with a rubber band"

Kitchen Tool Physical Commonsense Question: "Do I have to put on oven mitts while taking out the turkey from the microwave after defrosting"
Answer: "If it is not too hot, you could also use a paper towel or wait for it too cool down"


2. Temporal Commonsense Reasoning Questions :
Description : A temporal reasoning question is one which involves an inherent understanding of time in order to come up with an answer. To this effect we have identified 3 key dimensions of temporal reasoning and illustrate each with an example

Duration Temporal Commonsense Question: "Can I quickly go grab bread from the nearby store while the Turkey is still in the oven?"
Answer: Yes, you can quickly grab bread while the turkey is in the oven, provided it's safe and the oven is reliable. Ensure the turkey has enough cooking time left, set a timer, and if possible, let someone know to keep an eye on it while you step out.

Ordering Temporal Commonsense Question: "Can I put the eggs to boil first, so they’re done by the time I peel vegetables?"
Answer: Sure, you can boil eggs first and peel vegetables while they cook. Start the eggs in cold water, boil for 9-12 minutes, then cool them in ice water.

Frequency Temporal Commonsense Question: "How often should I stir the pot while making soup?"
Answer: Stir occasionally, especially if they contain ingredients like rice or pasta that might stick to the bottom of the pot. Stirring helps distribute heat evenly and prevents sticking and scorching.
```

LangChain's `ChatPromptTemplate`, `StructuredOutputParser` and `ResponseSchema` are used to organize the prompt input and outputs. For each QA turn in the task dialogue we prompt gpt to add crucial details by providing it the task instructions and the original QA from the dataset. We also generate follow-up question answer pairs to add clarifications and create data for holistic understanding of the task steps. We provide a sample input and output as one-shot prompt to the model.
Following is a prompt snippet used in dataset generation : 
```
prompt_preface =
"""
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


output_template =
"""
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
```













[^1]: @Inproceedings{Choi2022,
 author = {Jason Choi and Saar Kuzi and Nikhita Vedula and Jie Zhao and Giuseppe Castellucci and Marcus Collins and Shervin Malmasi and Oleg Rokhlenko and Eugene Agichtein},
 title = {Wizard of tasks: A novel conversational dataset for solving real-world tasks in conversational settings},
 year = {2022},
 url = {https://www.amazon.science/publications/wizard-of-tasks-a-novel-conversational-dataset-for-solving-real-world-tasks-in-conversational-settings},
 booktitle = {COLING 2022},
}
