import json, os
import random
import argparse

import torch
import torch.nn.functional as F
from tqdm import trange

from data import load_data_instances, DataIterator
from model import MultiInferxlnet
import spacy
from spacy.lang.en import English
import networkx as nx
import matplotlib.pyplot as plt
import utils
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

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
        #print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1],
                                                                  #aspect_results[2]))
        #print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1],
                                                                   #opinion_results[2]))
        #print(args.task + '\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1

def getSentences(text):
    nlp = English()
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    document = nlp(text)
    return [sent.string.strip() for sent in document.sents]


def appendChunk(original, chunk):
    return original + ' ' + chunk

def isRelationCandidate(token):
    deps = ["ROOT", "adj", "attr", "agent", "amod"]
    return any(subs in token.dep_ for subs in deps)

def test(args):
    print("Evaluation on testset:")
    model_path = args.model_dir + 'xlnet' + args.task + args.dataset +'.pt'
    config = XLNetConfig.from_pretrained('xlnet-base-cased', output_hidden_states=True)
    xlnet = XLNetModel.from_pretrained(args.xlnet_model_path,config=config)
    tokenizer = XLNetTokenizer.from_pretrained(args.xlnet_tokenizer_path)
    #model = torch.load(model_path, map_location=torch.device("cpu"))
    model = torch.load(model_path).to(args.device)
    
    model.eval()

    sentence_packs = json.load(open(args.prefix + args.dataset + '/test.json'))
    instances = load_data_instances(sentence_packs, args)
    testset = DataIterator(instances, args)
    eval(model, testset, args)

def XLnet_eval_test(tokens):
    subject = ''
    object = ''
    relation = ''
    subjectConstruction = ''
    objectConstruction = ''
    for token in tokens:
        #printToken(token)
        if "punct" in token.dep_:
            continue
        if isRelationCandidate(token):
            relation = appendChunk(relation, token.lemma_)
        if isConstructionCandidate(token):
            if subjectConstruction:
                subjectConstruction = appendChunk(subjectConstruction, token.text)
            if objectConstruction:
                objectConstruction = appendChunk(objectConstruction, token.text)
        if "subj" in token.dep_:
            subject = appendChunk(subject, token.text)
            subject = appendChunk(subjectConstruction, subject)
            subjectConstruction = ''
        if "obj" in token.dep_:
            object = appendChunk(object, token.text)
            object = appendChunk(objectConstruction, object)
            objectConstruction = ''

    print (subject.strip(), ",", relation.strip(), ",", object.strip())
    return (subject.strip(), relation.strip(), object.strip())

def printToken(token):
    print(token.text, "->", token.dep_)

def appendChunk(original, chunk):
    return original + ' ' + chunk

def processSentence(sentence):
    tokens = nlp_model(sentence)
    return XLnet_eval_test(tokens)

def isConstructionCandidate(token):
    deps = ["compound", "prep", "conj", "mod"]
    return any(subs in token.dep_ for subs in deps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel1/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="pair", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--dataset', type=str, default="res14", choices=["res14", "lap14", "res15", "res16"],
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--xlnet_model_path', type=str,
                        default="pretrained/",
                        help='pretrained bert model path')
    parser.add_argument('--xlnet_tokenizer_path', type=str,
                        default="xlnet-base-cased",
                        help='pretrained bert tokenizer path')
    parser.add_argument('--xlnet_feature_dim', type=int, default=768,
                        help='dimension of pretrained bert feature')

    parser.add_argument('--nhops', type=int, default=1,
                        help='inference times')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=120,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=4,
                        help='label number')

    args = parser.parse_args()

text = "The Continental cuisines are served by the waitress is taking so long"

sentences = getSentences(text)
nlp_model = spacy.load('en_core_web_sm')

triples = []
    #print (text)
for sentence in sentences:
    triples.append(processSentence(sentence))

