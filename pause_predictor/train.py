import random
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

from utils import get_data_from_xml, get_all_filepath
from data_utils import pauseCollator, pauseDataset

def main():
    # data preprocessing
    dir_path = "/mnt/aoni04/hsieh/topics/"
    xml_file_path = get_all_filepath(dir_path)
    # get all data
    id = 0
    all_data = []
    for xml_file in xml_file_path:
        id, data = get_data_from_xml(id, xml_file)
        all_data += data
    df = pd.DataFrame(all_data)
    # print(df.head())

    # split train, eval, test
    random.seed(10)
    train_df, eval_df = train_test_split(df, train_size=0.7)
    eval_df, test_df = train_test_split(eval_df, train_size=0.5)
    # print('train size', train_df.shape)
    # print('eval size', eval_df.shape)
    # print('test size', test_df.shape)

    # create dataset
    train_dataset = pauseDataset(train_df)
    eval_dataset = pauseDataset(eval_df)
    test_dataset = pauseDataset(test_df)
    print(train_dataset[10])
        
    # create collator
    tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    pause_collator = pauseCollator("clause_pause_classifier",tokenizer)




if __name__ == "__main__":
    main()

    