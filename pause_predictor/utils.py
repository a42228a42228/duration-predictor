import xml.etree.ElementTree as ET
from os import walk
from os.path import join

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