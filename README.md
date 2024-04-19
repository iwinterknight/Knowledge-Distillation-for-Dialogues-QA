# Knowledge-Distillation-for-Dialogues-QA

This repo contains :
1. Code to build a specialist LLM for Cooking tasks, fine-tuned with knowledge distillation from an open-source generalist LLM's (GPT 3.5 Turbo) predictions.
2. Code to augment Amazon's Wizard of Tasks[^1] dataset by generating more elaborate Question Answer turns for the recipes.
3. Retrieval Augmented Generation(RAG) pipelines using `Parent Document Retriever` and `Contextual Compression Retriever`












[^1]: @Inproceedings{Choi2022,
 author = {Jason Choi and Saar Kuzi and Nikhita Vedula and Jie Zhao and Giuseppe Castellucci and Marcus Collins and Shervin Malmasi and Oleg Rokhlenko and Eugene Agichtein},
 title = {Wizard of tasks: A novel conversational dataset for solving real-world tasks in conversational settings},
 year = {2022},
 url = {https://www.amazon.science/publications/wizard-of-tasks-a-novel-conversational-dataset-for-solving-real-world-tasks-in-conversational-settings},
 booktitle = {COLING 2022},
}
