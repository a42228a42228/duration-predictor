from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class pauseDataset(Dataset):
    def __init__(self, df):
        # define attribute
        self.features = [
            {
                'text': row.text,
                'clause_pause_list': row.clause_pause_list,
                'sentence_pause': row.sentence_pause,
            } for row in tqdm(df.itertuples(), total=df.shape[0])
        ]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]

class pauseCollator():
  def __init__(self, collator_type, tokenizer, max_length=512):
    self.collator_type = collator_type
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __call__(self, examples):
    if self.collator_type == "clause_pause_classifier":
      examples = {
              'text': list(map(lambda x: x['text'], examples)), # map(func,list):apply func in each item in list, and return to iterable map object
              'label': list(map(lambda x: x['clause_pause_list'], examples))
            }
      encodings = self.tokenizer(examples['text'],padding='longest',return_tensors='pt')
      encodings['label'] = torch.tensor([
                          self._align_labels_with_tokens(pause_list, input_ids)
                          for pause_list, input_ids in zip(examples['label'], encodings['input_ids'])
                        ])
      encodings = self._remove_underbar(encodings)
      encodings['label'] = torch.tensor([
                    self._make_binary_classifier_label(label) for label in encodings['label']
                  ])

    elif self.collator_type == "clause_pause_regression":
      examples = {
              'text': list(map(lambda x: x['text'], examples)), # map(func,list):apply func in each item in list, and return to iterable map object
              'label': list(map(lambda x: x['clause_pause_list'], examples))
            }
      encodings = self.tokenizer(examples['text'],padding='longest',return_tensors='pt')
      encodings['label'] = torch.tensor([
                          self._align_labels_with_tokens(pause_list, input_ids)
                          for pause_list, input_ids in zip(examples['label'], encodings['input_ids'])
                        ])
      encodings = self._remove_underbar(encodings)
      encodings['label'] = torch.tensor([
                    self._make_regression_label(label) for label in encodings['label']
                  ])

    elif self.collator_type == "sentence_pause":
      examples = {
        'text': list(map(lambda x: x['text'], examples)), # map(func,list):apply func in each item in list, and return to iterable map object
        'label': list(map(lambda x: x['sentence_pause'], examples))
      }

    return encodings

  def _make_binary_classifier_label(self, clause_pauses):
    new_label = []
    ignore_flag = -100
    for clause_pause in clause_pauses:
      if clause_pause == 0:
        new_label.append(0)
      elif clause_pause > 0:
        new_label.append(1)
      else:
        new_label.append(ignore_flag)
    return new_label

  def _make_regression_label(self, clause_pauses):
    new_label = []
    ignore_flag = -100
    for clause_pause in clause_pauses:
      if clause_pause > 0:
        new_label.append(clause_pause)
      else:
        new_label.append(ignore_flag)
    return new_label

  def _align_labels_with_tokens(self, labels, input_ids):
    new_labels = []
    label_idx = 0
    for input_id in input_ids:
      word = self.tokenizer.decode(input_id)
      if word == '|':
        if label_idx > len(labels) - 1:
          print(self.tokenizer.decode(input_ids))
          print(labels)
        new_labels.append(labels[label_idx])
        label_idx += 1
      else:
          new_labels.append(-100)
    return new_labels

  def _remove_underbar(self, encodings):
    new_input_ids = []
    new_attention_mask = []
    new_labels = []
    for input_id, attention_mask, label in zip(encodings['input_ids'], encodings['attention_mask'], encodings['label']):
      text_len = input_id.size(dim=0)

      # delete underbar
      while True:
        text = self.tokenizer.decode(input_id)
        # print(text)
        if '|' in text:
          for i, id in enumerate(input_id):
            word = self.tokenizer.decode(id)
            if word == '|':
              input_id = self._torch_del(input_id, i)
              attention_mask = self._torch_del(attention_mask, i)
              label = self._torch_del(label, i-1)
              break
        else:
          break
      new_text_len = input_id.size(dim=0)

      # add padding
      padding_size = text_len - new_text_len
      new_input_ids.append(torch.cat((input_id, torch.tensor([0] * padding_size)), dim=0))
      new_attention_mask.append(torch.cat((attention_mask, torch.tensor([0] * padding_size)), dim=0))
      new_labels.append(torch.cat((label, torch.tensor([-100] * padding_size)), dim=0))

    new_encodings = {"input_ids": torch.stack(new_input_ids), "token_type_ids": encodings['token_type_ids'], "attention_mask": torch.stack(new_attention_mask), "label": torch.stack(new_labels)}
    return new_encodings

  def _torch_del(self, tensor, idx):
    tensor1 = tensor[0:idx]
    tensor2 = tensor[idx+1:]
    return torch.cat((tensor1, tensor2), dim=0)