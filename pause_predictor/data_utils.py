import xml.etree.ElementTree as ET
import pandas as pd
from os import walk
from os.path import join
from torch.utils.data import Dataset
from tqdm import tqdm

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


import torch
from transformers import AutoTokenizer

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
      word = tokenizer.decode(input_id)
      if word == '|':
        if label_idx > len(labels) - 1:
          print(tokenizer.decode(input_ids))
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
        text = tokenizer.decode(input_id)
        # print(text)
        if '|' in text:
          for i, id in enumerate(input_id):
            word = tokenizer.decode(id)
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

tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
pause_collator = pauseCollator("clause_pause_classifier",tokenizer)



def get_data_from_xml(id, xml_file):
  tree = ET.parse(xml_file)
  root = tree.getroot()
  article = root[0]
  ignore_flag = -100
  data = []
  for i, sent in enumerate(article):
    text = ""
    clause_pause_list = [] # 文節ポーズ
    for j, phrase in enumerate(sent):
      text += phrase.text.replace("。", "")
      # ポーズ計算
      if i == len(article) - 1 and j == len(sent) - 1:    # 最後の文の最後の文末ポーズはロスに取り入れない
        pause = ignore_flag
        sentence_pause = ignore_flag
        # clause_pause_list += [ignore_flag] * len(phrase.text)
      else:
        if j == len(sent) - 1:  # 文末
          next_sent = article[i + 1]
          next_phrase = next_sent[0]
          pause = (int(next_phrase.attrib['start_time']) - int(phrase.attrib['end_time'])) * pow(10, -3)
          sentence_pause = pause
          # clause_pause_list += [ignore_flag] * len(phrase.text)
        else: #文節
          text += "_"
          next_phrase = sent[j + 1]
          pause = (int(next_phrase.attrib['start_time']) - int(phrase.attrib['end_time'])) * pow(10, -3)
          # clause_pause_list += [ignore_flag] * (len(phrase.text)-1) + [pause]
          clause_pause_list.append(pause)
    id += 1
    data += [{"id": id, "text": text, "clause_pause_list": clause_pause_list, "sentence_pause":sentence_pause, "file_path": xml_file}]
  return id, data

def get_all_filepath(dir_path):
    file_path = []
    for root, _, files in walk(dir_path):
        for f in files:
            if ".xml" in f:
                file_path.append(join(root, f))
    return file_path




dir_path = "/mnt/aoni04/hsieh/topics/"
# dir_path = "/content/drive/MyDrive/topics/" # for google colab
xml_file_path = get_all_filepath(dir_path)

# get all data
id = 0
all_data = []
for xml_file in xml_file_path:
    id, data = get_data_from_xml(id, xml_file)
    all_data += data

# show data
df = pd.DataFrame(all_data)
df.head()