import json, os
import random
import argparse
import re
from flask_ngrok import run_with_ngrok
from flask import Flask, render_template,url_for,request,redirect
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator
from model import MultiInferBert
import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
import utils
from transformers import BertModel, BertTokenizer ,BertConfig

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/')
def home():
	return render_template('home.html')

def train(args):

    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))
    instances_train = load_data_instances(train_sentence_packs, args)
    instances_dev = load_data_instances(dev_sentence_packs, args)
    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model = MultiInferBert(args).to(args.device)

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': 5e-5},
        {'params': model.cls_linear.parameters()}
    ], lr=5e-5)

    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(args.epochs):
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):
            _, tokens, lengths, masks, _, _, aspect_tags, tags = trainset.get_batch(j)
            preds = model(tokens, masks)

            preds_flatten = preds.reshape([-1, preds.shape[3]])
            tags_flatten = tags.reshape([-1])
            loss = F.cross_entropy(preds_flatten, tags_flatten, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = eval(model, devset, args)

        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + 'bert' + args.task +'.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))


def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_preds = []
        all_labels = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        for i in range(dataset.batch_count):
            sentence_ids, tokens, lengths, masks, sens_lens, token_ranges, aspect_tags, tags = dataset.get_batch(i)
            preds = model(tokens, masks)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_labels.append(tags)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)
            all_ids.extend(sentence_ids)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_labels = torch.cat(all_labels, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        metric = utils.Metric(args, all_preds, all_labels, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        precision, recall, f1 = metric.score_uniontags()
        aspect_results = metric.score_aspect()
        opinion_results = metric.score_opinion()
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   opinion_results[2]))
        print(args.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1

def test(args):
    print("Evaluation on testset:")
    model_path = args.model_dir + 'bert' + args.task + args.dataset + '1' +'.pt'
    model = torch.load(model_path).to(args.device)
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    instances = load_data_instances(sentence_packs, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args)

def predict(message1,args):
    print("mode: predict")
    model_path = args.model_dir + 'bert' + args.task +'.pt'
    config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
    bert = BertModel.from_pretrained(args.bert_model_path,config=config)
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer_path)
    model = torch.load(model_path).to(args.device)
    model.eval()
    triples = []
    triplets_sentences = []
    sentence = message1
    id = random.randint(1,1000)
    Aspect_spans = '\\O '.join(re.split("\s|[,.?!]",sentence))
    Opinion_spans = '\\O '.join(re.split("\s|[,.?!]",sentence))
    uid = random.randint(10000,99999)
    triples = [{"uid":uid,"target_tags":Aspect_spans,"opinion_tags":Opinion_spans}]
    triplets_sentences = [{"id":id,"sentence":sentence,"triples":triples}]
    with open("/content/drive/MyDrive/GTS-main/data/wine226/test3.json", "w+") as test_json:
      json.dump(triplets_sentences,test_json )

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test3.json'))
    instances = load_data_instances(sentence_packs, args)
    predictset = DataIterator(instances, args)

    
    with torch.no_grad():
        all_ids = []
        all_sentences = []
        all_preds = []
        all_lengths = []
        all_sens_lengths = []
        all_token_ranges = []
        for i in range(predictset.batch_count):
            sentence_ids,sentence, tokens, lengths, masks, sens_lens, token_ranges, _, _ = predictset.get_batch(i)
            preds = model(tokens, masks)
            preds = torch.argmax(preds, dim=3)
            all_preds.append(preds)
            all_lengths.append(lengths)
            all_sens_lengths.extend(sens_lens)
            all_token_ranges.extend(token_ranges)
            all_ids.extend(sentence_ids)
            all_sentences.extend(sentence)

        all_preds = torch.cat(all_preds, dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()

        predictor = utils.Predict(args, all_ids, all_sentences, all_preds, all_lengths, all_sens_lengths, all_token_ranges, ignore_index=-1)
        predicted_results = predictor.predict_uniontags()
        print("--------prediction of Triplet Start---------")
        print(predicted_results)
        print("-------------prediction of Triplet Ends-------")
    return predicted_results

@app.route('/predict1',methods=['POST'])
def predict1():
  if request.method == 'POST':
    message1 = request.form['message']
    message2 = predict(message1,args)
    return render_template('result.html',prediction = message2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="pair", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test","predict1"],
                        help='option: train, test,predict')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16","wine226"],
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--bert_model_path', type=str,
                        default="pretrained/bert-base-uncased",
                        help='pretrained bert model path')
    parser.add_argument('--bert_tokenizer_path', type=str,
                        default="bert-base-uncased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--bert_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')

    parser.add_argument('--nhops', type=int, default=1,
                        help='inference times')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=120,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=4,
                        help='label number')

    args = parser.parse_args()

    if args.task == 'triplet':
        args.class_num = 6

    if args.mode == 'train':
        train(args)
        test(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'predict1':
        app.run()
        predict1()

