import configparser

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

print(f"pytorch lightning version : {pl.__version__}")

from transformers import (
    T5TokenizerFast as T5Tokenizer
)

from definitions import *

config = configparser.ConfigParser()
config.read(os.path.join(get_config_path()))

from create_dataset import prepare_dataset_for_distillation


class KnowledgeDistillationDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data: pd.DataFrame,
            source_max_token_len: int = 128,
            student_max_token_len: int = 512,
            teacher_max_token_len: int = 512
    ):
        super().__init__()
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.student_max_token_len = student_max_token_len
        self.teacher_max_token_len = teacher_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        source_encoding = self.tokenizer(
            data_row["question"],
            data_row["recipe"],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        student_encoding = self.tokenizer(
            data_row["student_answer"],
            max_length=self.student_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        student_labels = student_encoding["input_ids"]
        student_labels[student_labels == 0] = -100

        teacher_encoding = self.tokenizer(
            data_row["teacher_answer"],
            max_length=self.teacher_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        teacher_labels = teacher_encoding["input_ids"]
        teacher_labels[teacher_labels == 0] = -100

        return dict(
            question=data_row["question"],
            context=data_row["recipe"],
            student_answer=data_row["student_answer"],
            teacher_answer=data_row["teacher_answer"],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            student_labels=student_labels.flatten(),
            teacher_labels=teacher_labels.flatten()
        )


class KnowledgeDistillationDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer,
            train_df: pd.DataFrame,
            validation_df: pd.DataFrame,
            batch_size: int = 816,
            source_max_token_len: int = 128,
            student_max_token_len: int = 512,
            teacher_max_token_len: int = 512
    ):
        super().__init__()
        self.train_df = train_df
        self.validation_df = validation_df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.student_max_token_len = student_max_token_len
        self.teacher_max_token_len = teacher_max_token_len

    def setup(self, stage=None):
        self.train_dataset = KnowledgeDistillationDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.student_max_token_len,
            self.teacher_max_token_len
        )

        self.validation_dataset = KnowledgeDistillationDataset(
            self.validation_df,
            self.tokenizer,
            self.source_max_token_len,
            self.student_max_token_len,
            self.teacher_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )


def create_distillation_dataloader(model_name):
    train_df, validation_df = prepare_dataset_for_distillation()
    batch_size = config['training']['batch_size']
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    data_module = KnowledgeDistillationDataModule(train_df, validation_df, tokenizer, batch_size=batch_size)
    data_module.setup()
    return data_module