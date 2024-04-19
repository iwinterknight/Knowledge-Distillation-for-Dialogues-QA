# Knowledge-Distillation-for-Dialogues-QA

This repo contains :
1. Code to build a specialist LLM for Cooking tasks, fine-tuned with knowledge distillation from an open-source generalist LLM's (GPT 3.5 Turbo) predictions.
2. Code to augment Amazon's Wizard of Tasks[^1] dataset by generating more elaborate Question Answer turns for the recipes.
3. Retrieval Augmented Generation(RAG) pipelines using `Parent Document Retriever` and `Contextual Compression Retriever` and TruLens evaluation for the 3 RAG Triad evaluations : context relevance, groundedness and answer relevance.



**Knowledge Distillation**
Instruction fine-tuned LLMs are capable of in-context learning and generalize well to tasks they have not encountered during the fine-tuning stage. However this capacity to generalize is seen when the order of parameters is large.  

<p align="center">
 <img width="789" alt="RLHF Model Architecture" src=https://github.com/iwinterknight/Knowledge-Distillation-for-Dialogues-QA/assets/37212007/a501eb92-a098-4516-8b21-6545e4ae2bf3>
 <em>Image Source : [here] (https://arxiv.org/abs/2206.07682)</em>
</p>










[^1]: @Inproceedings{Choi2022,
 author = {Jason Choi and Saar Kuzi and Nikhita Vedula and Jie Zhao and Giuseppe Castellucci and Marcus Collins and Shervin Malmasi and Oleg Rokhlenko and Eugene Agichtein},
 title = {Wizard of tasks: A novel conversational dataset for solving real-world tasks in conversational settings},
 year = {2022},
 url = {https://www.amazon.science/publications/wizard-of-tasks-a-novel-conversational-dataset-for-solving-real-world-tasks-in-conversational-settings},
 booktitle = {COLING 2022},
}
