import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import XLMRobertaConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig, get_linear_schedule_with_warmup

from ignite.engine.engine import Engine, State, Events
from ignite.utils import convert_tensor
from ignite.metrics import Accuracy, Loss


from tqdm import tqdm


import json, nltk
from datetime import datetime
import re
import os
import string

#from transformers import file_utils
#print(file_utils.default_cache_path)
#exit()

app_configs = {}

class PreprocessDataseNew(Dataset):
    def __init__(self, df, tokenizer, max_len = 512):
        
        self.tokenizer = tokenizer
        self.texts = df.text.values
        self.labels = df.label.values
        self.max_len = max_len
        
    def __len__(self):
        
        return len(self.texts)
    
    def __getitem__(self,idx):
        
        tokens,mask,tokens_len = self.get_token_mask(self.texts[idx],self.max_len)
        label = self.labels[idx]
        return [torch.tensor(tokens),torch.tensor(mask),torch.tensor(tokens_len)],label

        
    def get_token_mask(self,text,max_len):
        
        tokens = []
        mask = []
        text = self.tokenizer.encode(text)
        size = len(text)
        
        pads = []
        if max_len - size > 0:
            pads = self.tokenizer.encode(['[PAD]'] * max(0, max_len - size), add_special_tokens=False)
            tokens[:max(max_len,size)] = text[:max(max_len,size)]
            tokens = tokens + pads[1:-1]
        mask = [1]*size+[0]*len(pads[1:-1])
        print("mask=", mask)
        exit()
        tokens_len = len(tokens)
        
        return tokens,mask,tokens_len
    

class PreprocessDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        texts = dataframe.text.values.tolist()

        texts = [self._preprocess(text) for text in texts]

        #self._print_random_samples(texts)

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=150,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        if 'label' in dataframe:
            self.labels = dataframe.label.values.tolist()

    def _print_random_samples(self, texts):
        np.random.seed(42)
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")

        print()

    def _preprocess(self, text):
        original_text = text
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_hashes(text)
        text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)

        #text = self._lowercase(text)
        text = self._remove_punctuation(text)
        #text = self._remove_numbers(text)

        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        #text_tokens = self._stemming(text_tokens)
        #text_tokens = self._lemmatisation(text_tokens)
        
        text = self._stitch_text_tokens_together(text_tokens)
        text = text.strip()
        return text;


    def _remove_amp(self, text):
        return text.replace("&amp;", " ")

    def _remove_mentions(self, text):
        return re.sub(r'(@.*?)[\s]', ' ', text)
    
    def _remove_multiple_spaces(self, text):
        return re.sub(r'\s+', ' ', text)

    def _remove_retweets(self, text):
        return re.sub(r'^RT[\s]+', ' ', text)

    def _remove_links(self, text):
        return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

    def _remove_hashes(self, text):
        return re.sub(r'#', ' ', text)

    def _stitch_text_tokens_together(self, text_tokens):
        return " ".join(text_tokens)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language="english")

    def _stopword_filtering(self, text_tokens):
        stop_words = nltk.corpus.stopwords.words('english')

        return [token for token in text_tokens if token not in stop_words]

    # Stemming (remove -ing, -ly, ...)
    def _stemming(self, text_tokens):
        porter = nltk.stem.porter.PorterStemmer()
        return [porter.stem(token) for token in text_tokens]

    # Lemmatisation (convert the word into root word)
    def _lemmatisation(self, text_tokens):
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        return [lem.lemmatize(token) for token in text_tokens]
    
    def _remove_numbers(self, text):
        return re.sub(r'\d+', ' ', text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, text):
        return ''.join(character for character in text if character not in string.punctuation)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label
        
class CustomClassifierRobertaLarge(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifierRobertaLarge, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(1024, 8)
        self.fc2 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
    
# Definirea modelului BERT
class LSTMClassifier(nn.Module):
    def __init__(self, pretrained_model, num_labels=1):
        super(LSTMClassifier, self).__init__()
        
        self.bert = pretrained_model
        self.hidden_size = self.bert.config.hidden_size
        lstm_hidden_size = self.hidden_size
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=lstm_hidden_size,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(2*lstm_hidden_size, num_labels) #2 times because is bidirectional
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # Obțineți starea ascunsă de la BERT
        bert_output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state

        # Aplicați LSTM peste starea ascunsă de la BERT
        lstm_output, _ = self.lstm(bert_output)

        # Luați output-ul ultimului strat de la LSTM și aplicați un strat fully connected pentru clasificare
        output = lstm_output[:, -1, :]
        output = self.fc(output)
        # Aplicați funcția de activare la ieșire
        output = self.sigmoid(output)

        return output
    
                   
def str_to_class(s):
    #if s in globals() and isinstance(globals()[s], types.ClassType):
    return globals()[s]
    

def target_device():
    #gpu support for Mac
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()        
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")    
    app_configs['device'] = device
    print("Use device:", device)
    return device

  
def train(model, learning_rate, epochs, model_name):        
    best_val_loss = float('inf')
    best_val_accuracy = float('-inf')
    early_stopping_threshold_count = 0
    
    device = app_configs['device']
    
    if (app_configs['percent_of_data'] >= 50):
        train_dataloader, val_dataloader = load_and_preprocess_data()
        
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    
    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        model.train()
        if (app_configs['percent_of_data'] < 50):
            train_dataloader, val_dataloader = load_and_preprocess_data()
        
        for train_input, train_label in tqdm(train_dataloader):
            optimizer.zero_grad()
        
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            #print("for train size=", input_ids.shape, attention_mask.shape)    
            train_label = train_label.to(device)

            output = model(input_ids, attention_mask)

            loss = criterion(output, train_label.float().unsqueeze(1))

            total_loss_train += loss.item()

            acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            

        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0
            
            model.eval()
            
            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)

                loss = criterion(output, val_label.float().unsqueeze(1))

                total_loss_val += loss.item()

                acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc
            
            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
            
            if (best_val_loss > total_loss_val):
                best_val_loss = total_loss_val
                best_val_accuracy = total_acc_val
                torch.save(model, app_configs['models_path'] + model_name + ".pt")
                print("Saved model due to loss")
                early_stopping_threshold_count = 0
            elif (best_val_accuracy < total_acc_val):
                best_val_accuracy = total_acc_val
                torch.save(model, app_configs['models_path'] + model_name + ".pt")
                print("Saved model due to accuracy")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1
                
            #if early_stopping_threshold_count >= 1:
            #    print("Early stopping")
            #    break
            
def get_text_predictions(model, loader):
    device = app_configs['device']
    model = model.to(device)
    
    results_predictions = []
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)


            output = model(input_ids, attention_mask)
            
            output = (output > 0.5).int()
            results_predictions.append(output)
    
    return torch.cat(results_predictions).cpu().detach().numpy()
    
def get_pretrained_model():
    global app_configs
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
    
    if (app_configs['base_model'] == 'xlm-roberta-base'):
        print("load xlm")
        exit()
        pretrained_model = AutoModelForMaskedLM.from_pretrained(app_configs['base_model'])
    else:
        pretrained_model = AutoModel.from_pretrained(app_configs['base_model'])
        
    app_configs['tokenizer'] = tokenizer
    app_configs['pretrained_model'] = pretrained_model
    return tokenizer, pretrained_model
    
  
def get_train_data(train_path, random_seed = 0):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    

    percentOfData = app_configs['percent_of_data']
    train_df = train_df.sample(int(len(train_df)*percentOfData/100))
    print("train len=", len(train_df))
    return train_df    

def get_val_data(val_path, random_seed = 0):
    """
    function to read dataframe with columns
    """

    val_df = pd.read_json(val_path, lines=True)
    

    percentOfData = app_configs['percent_of_data']
    val_df = val_df.sample(int(len(val_df)*percentOfData/100))
    print("val len=", len(val_df))
    return val_df    

def load_and_preprocess_data(split=0):
    tokenizer = app_configs['tokenizer']
    # Load JSON file with dataset. Perform basic transformations.
    train_df = get_train_data(app_configs['train_path'])    
    
    if (split == 1):
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
    else:
        val_df = get_val_data(app_configs['val_path'])    
            
    train_df = train_df.drop(["model", "source"], axis=1)
    val_df = val_df.drop(["model", "source"], axis=1)
    train_dataloader = DataLoader(PreprocessDataset(train_df, tokenizer), batch_size=8, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(PreprocessDataset(val_df, tokenizer), batch_size=8, num_workers=0)
    return train_dataloader, val_dataloader

def create_and_train():
    target_device()    
    tokenizer, pretrained_model = get_pretrained_model()
        
    
    
    classifierClass = str_to_class(app_configs['classifier'])
    myModel = classifierClass(pretrained_model)
    train(myModel, learning_rate=app_configs['learning_rate'], epochs=app_configs['epochs'], model_name=app_configs['model_name'])
    
def load_and_evaluate(model_name = ''):
    #global app_configs
    if (model_name):
        app_configs['model_name'] = model_name
    
    test_df = pd.read_json(app_configs['test_path'], lines=True)
    if 'model' in test_df:
        test_df = test_df.drop(["model"], axis=1)
    if 'source' in test_df:
        test_df = test_df.drop(["source"], axis=1)
    
    target_device()
    tokenizer, pretrained_model = get_pretrained_model()
    
    model_path = app_configs['models_path'] + app_configs['model_name'] + ".pt"    
    model = torch.load(model_path)
    print("model loaded:", model_path);
    predictions_df = pd.DataFrame({'id': test_df['id']})
    test_dataloader = DataLoader(PreprocessDataset(test_df, tokenizer), batch_size=8, shuffle=False, num_workers=0)
    predictions_df['label'] = get_text_predictions(model, test_dataloader)
    #
    predictions_df.to_json(app_configs['prediction_path'], lines=True, orient='records')
    merged_df = predictions_df.merge(test_df, on='id', suffixes=('_pred', '_gold'))
    accuracy = accuracy_score(merged_df['label_gold'], merged_df['label_pred'])
    app_configs['accuracy'] = accuracy
    print("Accuracy:", accuracy)
    print(classification_report(merged_df['label_gold'], merged_df['label_pred']))
    if (model_name == ''): #save app options only when evaluation is called right after training
        save_app_options()


def load_app_options(model_name):
    options = pd.read_json(train_path, lines=True)
    
def save_app_options():
    configs = app_configs.copy()
    configs_keys = configs.keys()
    
    keys_2_del = {'tokenizer', 'pretrained_model', 'prediction_path', 'results_path', 'options_path', 'options_path', 'classifier', 'device'}
    for del_key in keys_2_del:
        configs.pop(del_key, None)
           
        
    # Writing to sample.json
    with open(app_configs['options_path'], "w") as outfile:
        json.dump(configs, outfile)
        
# datetime object containing current date and time
start_now = datetime.now()
start_time= start_now.strftime("%Y-%m-%d %H-%M")
timestamp_prefix = start_now.strftime("%Y%m%d%H%M")
print("process start at:", start_time)
torch.manual_seed(0)
np.random.seed(0)

absolute_path = os.path.abspath('data')


distilbert_model_configs1 = {
    'base_model': 'distilbert-base-uncased',
    'classifier': 'CustomClassifierBase',
}

distilbert_model_configs2 = {
    'base_model': 'distilbert-base-uncased',
    'classifier': 'CustomClassifierBase3Layers',
    'learning_rate': 2e-5,
}

robertabase_model_configs1 = {
    'base_model': 'roberta-base',
    'classifier': 'CustomClassifierBase', 
}

robertalarge_model_configs1 = {
    'base_model': 'roberta-large',    
    #'classifier': 'CustomClassifierRobertaLarge', 
    'classifier': 'LSTMClassifier',     
    'percent_of_data': 5,
    'learning_rate': 1e-5,
}

bert_model_configs1 = {
    'base_model': 'bert-base-uncased',
    'classifier': 'CustomClassifierBase',
}

bert_model_configs2 = {
    'base_model': 'bert-base-uncased',    
    'classifier': 'LSTMClassifier',     
    'percent_of_data': 10,
    'learning_rate': 2e-5,
}

distilbert_model_configs2 = {
    'base_model': 'distilbert-base-uncased',
    'classifier': 'LSTMClassifier',     
    'percent_of_data': 10,
    'learning_rate': 1e-5,
}

albert_model_configs1 = {
    'base_model': 'albert-base-v2',
    'classifier': 'CustomClassifierAlbert',
}

robertamultilang_model_configs1 = {
    'base_model': 'FacebookAI/xlm-roberta-base',
    'classifier': 'CustomClassifierBase', 
    'train_path': absolute_path + '/subtaskA_train_multilingual.jsonl',
    'test_path': absolute_path + '/subtaskA_dev_multilingual.jsonl',
}

robertalargemultilang_model_configs1 = {
    'base_model': 'FacebookAI/xlm-roberta-large',    
    'classifier': 'CustomClassifierRobertaLarge', 
    'train_path': absolute_path + '/subtaskA_train_multilingual.jsonl',
    'test_path': absolute_path + '/subtaskA_dev_multilingual.jsonl',
}

distilbertmultilang_model_configs1 = {
    'base_model': 'distilbert-base-multilingual-cased',
    'classifier': 'CustomClassifierBase', 
    'train_path': absolute_path + '/subtaskA_train_multilingual.jsonl',
    'test_path': absolute_path + '/subtaskA_dev_multilingual.jsonl',
}

bertmultilang_model_configs1 = {
    'base_model': 'bert-base-multilingual-cased',
    'classifier': 'CustomClassifierBase', 
    'train_path': absolute_path + '/subtaskA_train_multilingual.jsonl',
    'test_path': absolute_path + '/subtaskA_dev_multilingual.jsonl',
}

default_configs = {
    'learning_rate': 1e-5,
    'epochs': 5,
    'task': 'subtaskA_monolingual',
    'timestamp_prefix': timestamp_prefix,
    'train_path': absolute_path + '/subtaskA_train_monolingual.jsonl',
    'val_path': absolute_path + '/subtaskA_dev_monolingual.jsonl',
    'test_path': absolute_path + '/subtaskA_gold_monolingual.jsonl',
    'percent_of_data': 100,
    'options_path': absolute_path + '/predictions/'  + 'tests.results.jsonl',
    'models_path':  absolute_path + '/models/',
}


app_configs = default_configs.copy()
app_configs.update(bert_model_configs2)

app_configs['model_name'] = app_configs['timestamp_prefix'] + "_" + app_configs['task'] + "_" + app_configs['base_model'].replace("/", "_")
app_configs['prediction_path'] = absolute_path + '/predictions/' + app_configs['model_name'] + '.predictions.jsonl'
app_configs['options_path'] = absolute_path + '/predictions/'  + app_configs['model_name'] + '.options.jsonl'
app_configs['results_path'] = absolute_path + '/predictions/'  + app_configs['model_name'] + '.results.jsonl'

print("Working on pretrained-model:", app_configs['base_model'])

#model names that can be used for evaluation:
#model name roberta-large trained = 202401112145_subtaskA_monolingual_roberta-large
#model name roberta-base trained = 202402061024_subtaskA_monolingual_roberta
#model name for distilbert-base-uncased trained = 202401120919_subtaskA_monolingual_distilbert-base-uncased - 2 layers

#multilang tests
#model name for xlm_roberta_base = 202401201729_subtaskA_multilingual_FacebookAI_xlm-roberta-base
#model name for distilbert-base-multilingual-cased = 202401231736_subtaskA_monolingual_distilbert-base-multilingual-cased.options.jsonl
#creare_train_evaluate_vectorised()
#exit()
#model_for_evaluate='202402061024_subtaskA_monolingual_roberta-base'
model_for_evaluate=''
create_and_train()
load_and_evaluate(model_for_evaluate)

end_now = datetime.now()
end_time = end_now.strftime("%Y-%m-%d %H-%M")
print("process finished at:", end_time)
running_time = (end_now - start_now).total_seconds()
app_configs['start_time'] = start_time
app_configs['end_time'] = end_time
app_configs['running_time'] = running_time
if (model_for_evaluate == ''): 
    save_app_options()
