import random
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch

from utils import get_all_data, get_all_filepath
from data_utils import pauseCollator, pauseDataset
from model import ClauseClassifer
from eval_metric import compute_metrics

def run():
    # data preprocessing
    dir_path = "/mnt/aoni04/hsieh/topics/"
    xml_file_path = get_all_filepath(dir_path)
    all_data = get_all_data(xml_file_path)
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
    pause_collator = pauseCollator("clause_pause_classifier", tokenizer)

    # define loss function
    loss_fct = nn.CrossEntropyLoss()
    
    # define model
    pretrained_model = AutoModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model = ClauseClassifer(pretrained_model, 2, loss_fct)
    # freeze parameter in BERT
    for param in model.bert.parameters():
        param.requires_grad = False

    # set model to GPU
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    
    # inputs = tokenizer("私は昨日、公園に行きました。", return_tensors="pt")
    # inputs.to(device)
    # label = torch.tensor([[0,0,1,0,0,0,0,-100,0,0,-100,0,0]]).to(device)
    # result = model(**inputs, label=label)
    # print(result)
    # result.logits = result.logits.detach().cpu().numpy()
    # # print(result.logits)
    # print(compute_metrics((result.logits, label)))
    # # result
    
    # set training arguments
    output_dir = "/mnt/aoni04/hsieh/topics/model/clause_pause_classifier/"
    args = TrainingArguments(
                                output_dir=output_dir,
                                evaluation_strategy="epoch",
                                logging_strategy='epoch',
                                save_strategy="epoch",
                                save_total_limit=1,
                                lr_scheduler_type='constant',
                                learning_rate=1e-5,
                                label_names='labels',
                                metric_for_best_model='f1',
                                num_train_epochs=20,
                                weight_decay=0.01,
                                remove_unused_columns=False,
                                report_to='wandb',
                            )
    
    # define trainer
    trainer = Trainer(
                        model=model,
                        args=args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        data_collator=pause_collator,
                        compute_metrics=compute_metrics,
                        tokenizer=tokenizer,
                    )
    
    # train
    trainer.train()
    
    # save model
    # trainer.save_state()
    # trainer.save_model()


def main():
    run()


if __name__ == "__main__":
    main()

    