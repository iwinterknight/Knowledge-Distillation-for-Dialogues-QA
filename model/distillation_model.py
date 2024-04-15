import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AdamW
)

import pytorch_lightning as pl

print(f"pytorch lightning version : {pl.__version__}")


class DistillerModel(pl.LightningModule):
    def __init__(self, peft_model, alpha=0.5, temperature=0.4):
        super().__init__()
        self.model = peft_model
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        student_labels = batch["student_labels"]
        teacher_labels = batch["teacher_labels"]
        student_loss, student_logits = self(input_ids, attention_mask, student_labels)
        teacher_loss, teacher_logits = self(input_ids, attention_mask, teacher_labels)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        distillation_loss = kl_loss(nn.functional.softmax(student_logits / self.temperature, dim=1),
                                    nn.functional.softmax(teacher_logits / self.temperature, dim=1))
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        student_labels = batch["student_labels"]
        teacher_labels = batch["teacher_labels"]
        student_loss, student_logits = self(input_ids, attention_mask, student_labels)
        teacher_loss, teacher_logits = self(input_ids, attention_mask, teacher_labels)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        distillation_loss = kl_loss(nn.functional.softmax(student_logits / self.temperature, dim=1),
                                    nn.functional.softmax(teacher_logits / self.temperature, dim=1))
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        student_labels = batch["student_labels"]
        teacher_labels = batch["teacher_labels"]
        student_loss, student_logits = self(input_ids, attention_mask, student_labels)
        teacher_loss, teacher_logits = self(input_ids, attention_mask, teacher_labels)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        distillation_loss = kl_loss(nn.functional.softmax(student_logits / self.temperature, dim=1),
                                    nn.functional.softmax(teacher_logits / self.temperature, dim=1))
        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)
