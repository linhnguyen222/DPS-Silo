# Imports
import pandas as pd
from classifiers.bert_classifier import BertClassifier
from classifiers.bert_custom_models import BertPoolType
from classifiers.bert_classifier import BertTrainConfig
from embedding.embeddings_service import TextSpan
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

# Helper functions

def inst2sents(inst):   
    text = inst["text"]
    sents = inst["sents"]
    sents = [text[s[0]:s[1]] for s in sents]
    return sents


def instance_split_with_rationales(inst,down_weight, up_weight=1):
    inst_text = inst["text"]
    inst_sentences = inst["sents"]
    inst_spans = inst["spans"] #rationales
    #print('\n'+inst_text)
    #print(inst_sentences)
    #print(inst_spans)
    #print(t_i_spans)
    inst_text_spans = []
    for s in inst_sentences:
        s_start = s[0]
        s_end = s[1]
        s_text = inst_text[s_start:s_end]
        s_weights = [down_weight] * len(s_text.split())
        #print(s_weights)
        for i_span in inst_spans:
            if s_start <= i_span[0] and s_end>=i_span[1] :
                token_start = len(inst_text[s_start:i_span[0]].split())
                token_end = token_start + len(inst_text[i_span[0] : i_span[1]].split())
                #print(token_start)
                #print(token_end)
                for i in range(token_start, token_end-1):
                    s_weights[i]=up_weight
        inst_text_spans.append(TextSpan(s_text.split(), weights=s_weights))
    return inst_text_spans


# Data

train_sample = pd.read_json('train.json')
train_sample.columns = ['text', 'sents', 'label', 'spans']

train_examples = [inst2sents(inst) for i, inst in train_sample.iterrows()]
train_labels = [inst["label"] for i, inst in train_sample.iterrows()]
# label_list = [0, 'Loaded_Language', 'Doubt', 'Causal_Oversimplification',
#        'Name_Calling,Labeling', 'Red_Herring',
#        'Exaggeration,Minimisation', 'Repetition',
#        'Obfuscation,Intentional_Vagueness,Confusion', 'Straw_Men',
#        'Flag-Waving', 'Thought-terminating_Cliches', 'Slogans',
#        'Whataboutism', 'Appeal_to_fear-prejudice', 'Appeal_to_Authority',
#        'Reductio_ad_hitlerum', 'Black-and-White_Fallacy', 'Bandwagon']

label_list = list(train_sample['label'].unique())
label2id = {label : i for i, label in enumerate(label_list)}

# label_counts = Counter(train_labels)
# total_count = sum(label_counts.values())
# label_weights = {label: total_count/freq for label, freq in label_counts.items()}
label_weights = {label_list[i]: compute_class_weight(class_weight='balanced', classes=label_list, y=train_labels)[i] for i in range(len(label_list))}

train_token_label_ids = [[text_span.weights for text_span in 
# multi-label rationales
instance_split_with_rationales(inst,down_weight=0, up_weight=1)] for i, inst in train_sample.iterrows()]

bert = BertClassifier(pretrained_bert='bert-base-uncased',
                      num_labels=len(label_list),
                      is_multi_label_rationales=False)

# AVG = 1, MAX = 2, RNN = 3
bert_pool_type = BertPoolType(1)

# Config
train_config = BertTrainConfig(num_train_epochs=7,
                               learning_rate=5e-6,
                               upper_dropout=0.1,
                               max_seq_length=256,
                               batch_size=32)


bert.train(bert_pool_type, train_examples, train_labels, label_list, label2id, label_weights,
           train_token_label_ids=train_token_label_ids,
           train_config=train_config,
           rationale_weight = 1.0,
           text_weight=1.0,
           two_berts=False,
           detach_weights=True,
           learn_weights=False,
           bert_independent_rationales=False,
           shallow_fine_tuning=False)


test_sample_prop = pd.read_json('test_prop.json')
test_examples = [inst2sents(inst) for i, inst in test_sample_prop.iterrows()]
test_predictions = bert.predict(test_examples)
true_labels = test_sample_prop['label'].values
out = pd.DataFrame([[true_labels[i] ,test_predictions[i][0]] for i in range(len(test_predictions))])
out.to_json('results_prop.json')



test_sample_own = pd.read_json('test_own.json')
test_examples = [inst2sents(inst) for i, inst in test_sample_own.iterrows()]
test_predictions = bert.predict(test_examples)
true_labels = test_sample_own['label'].values
out = pd.DataFrame([[true_labels[i] ,test_predictions[i][0]] for i in range(len(test_predictions))])
out.to_json('results_own.json')
