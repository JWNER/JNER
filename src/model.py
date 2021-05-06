import config
import torch
import transformers
import torch.nn as nn
import unicodedata

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_los = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_lables = torch.where(
        active_los,
        target.view(-1),
        # lfn = nn.CrossEntropyLoss()
        # lfn.ignore_index --> -100
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_lables)
    return loss


class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):
        # Transformers uses O1-level optimized mixed precision training implemented with the Apex library [paper.REVISITING FEW-SAMPLE BERT FINE-TUNING]
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        bo_tag = self.bert_drop_1(o1)
        bo_pos = self.bert_drop_2(o1)

        tag = self.out_tag(bo_tag)
        pos = self.out_pos(bo_pos)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)
        loss_pos = loss_fn(pos, target_pos, mask, self.num_pos)

        loss = (loss_tag + loss_pos) / 2

        return tag, pos, loss

