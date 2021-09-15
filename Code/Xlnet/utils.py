import multiprocessing
import pickle
import numpy as np
import sklearn

id2sentiment = {1: 'neg', 3: 'neu', 5: 'pos'}


def get_aspects(tags, length, ignore_index=-1):
    spans = []
    start = -1
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 1:
            if start == -1:
                start = i
        elif tags[i][i] != 1:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans


def get_opinions(tags, length, ignore_index=-1):
    spans = []
    start = -1
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 2:
            if start == -1:
                start = i
        elif tags[i][i] != 2:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length-1])
    return spans


class Metric():
    def __init__(self, args, predictions, goldens, xlnet_lengths, sen_lengths, tokens_ranges, ignore_index=-1):
        self.args = args
        self.predictions = predictions
        self.goldens = goldens
        self.xlnet_lengths = xlnet_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = -1
        self.data_num = len(self.predictions)

    def get_spans(self, tags, length, token_range, type):
        spans = []
        start = -1
        for i in range(length):
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_pair(self, tags, aspect_spans, opinion_spans, token_ranges):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 4
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if tag_num[3] == 0: continue
                sentiment = -1
                pairs.append([al, ar, pl, pr, sentiment])
        return pairs

    def find_triplet(self, tags, aspect_spans, opinion_spans, token_ranges):
        triplets = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 6
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                        # if tags[i][j] != -1:
                        #     tag_num[int(tags[i][j])] += 1
                        # if tags[j][i] != -1:
                        #     tag_num[int(tags[j][i])] += 1
                if sum(tag_num[3:]) == 0: continue
                sentiment = -1
                if tag_num[5] >= tag_num[4] and tag_num[5] >= tag_num[3]:
                    sentiment = 5
                elif tag_num[4] >= tag_num[3] and tag_num[4] >= tag_num[5]:
                    sentiment = 4
                elif tag_num[3] >= tag_num[5] and tag_num[3] >= tag_num[4]:
                    sentiment = 3
                if sentiment == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    input()
                triplets.append([al, ar, pl, pr, sentiment])
        return triplets

    def score_aspect(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            for spans in golden_aspect_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            for spans in predicted_aspect_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_opinion(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_opinion_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            for spans in golden_opinion_spans:
                golden_set.add(str(i) + '-' + '-'.join(map(str, spans)))

            predicted_opinion_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            for spans in predicted_opinion_spans:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, spans)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def score_uniontags(self):
        assert len(self.predictions) == len(self.goldens)
        golden_set = set()
        predicted_set = set()
        for i in range(self.data_num):
            golden_aspect_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            golden_opinion_spans = self.get_spans(self.goldens[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            if self.args.task == 'pair':
                golden_tuples = self.find_pair(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                golden_tuples = self.find_triplet(self.goldens[i], golden_aspect_spans, golden_opinion_spans, self.tokens_ranges[i])
            for pair in golden_tuples:
                golden_set.add(str(i) + '-' + '-'.join(map(str, pair)))

            predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            predicted_opinion_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            if self.args.task == 'pair':
                predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                predicted_tuples = self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            for pair in predicted_tuples:
                predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))

        correct_num = len(golden_set & predicted_set)
        precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
        recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

class Predict():
    def __init__(self, args, data_ids,data_sentences, predictions, xlnet_lengths, sen_lengths, tokens_ranges, ignore_index=-1):
        self.args = args
        self.data_ids = data_ids
        self.data_sentences = data_sentences
        self.predictions = predictions
        self.xlnet_lengths = xlnet_lengths
        self.sen_lengths = sen_lengths
        self.tokens_ranges = tokens_ranges
        self.ignore_index = -1
        self.data_num = len(self.predictions)

    def get_spans(self, tags, length, token_range, type):
        spans = []
        start = -1
        for i in range(length):
            l, r = token_range[i]
            if tags[l][l] == self.ignore_index:
                continue
            elif tags[l][l] == type:
                if start == -1:
                    start = i
            elif tags[l][l] != type:
                if start != -1:
                    spans.append([start, i - 1])
                    start = -1
        if start != -1:
            spans.append([start, length - 1])
        return spans

    def find_pair(self, tags, aspect_spans, opinion_spans, token_ranges):
        pairs = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 4
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                if tag_num[3] == 0: continue
                sentiment = "None"
                pairs.append([al, ar, pl, pr, sentiment])
                print
        return pairs

    def find_triplet(self, tags, aspect_spans, opinion_spans, token_ranges):
        triplets = []
        for al, ar in aspect_spans:
            for pl, pr in opinion_spans:
                tag_num = [0] * 6
                for i in range(al, ar + 1):
                    for j in range(pl, pr + 1):
                        a_start = token_ranges[i][0]
                        o_start = token_ranges[j][0]
                        if al < pl:
                            tag_num[int(tags[a_start][o_start])] += 1
                        else:
                            tag_num[int(tags[o_start][a_start])] += 1
                        # if tags[i][j] != -1:
                        #     tag_num[int(tags[i][j])] += 1
                        # if tags[j][i] != -1:
                        #     tag_num[int(tags[j][i])] += 1
                if sum(tag_num[3:]) == 0: continue
                sentiment = -1
                if tag_num[5] >= tag_num[4] and tag_num[5] >= tag_num[3]:
                    sentiment = 'positive'
                elif tag_num[4] >= tag_num[3] and tag_num[4] >= tag_num[5]:
                    sentiment = 'neutral'
                elif tag_num[3] >= tag_num[5] and tag_num[3] >= tag_num[4]:
                    sentiment = 'negative'
                if sentiment == -1:
                    print('wrong!!!!!!!!!!!!!!!!!!!!')
                    input()
                triplets.append([al, ar, pl, pr, sentiment])
        return triplets

    def predict_uniontags(self):
        predicted_results = dict()
        triplets_sentences = []
        aspect_tags = []
        opinion_tags = []
        sent = []
        aspect_no_tuple = []
        opinion_no_tuple = []
        result = []
        result1 = []
        triplets_sentences_no_opinion = []
        for i in range(self.data_num):
            predicted_aspect_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 1)
            list_sentences = (self.data_sentences[i]).split()
            raw_sentences = self.data_sentences[i]
            predicted_opinion_spans = self.get_spans(self.predictions[i], self.sen_lengths[i], self.tokens_ranges[i], 2)
            if self.args.task == 'pair':
                predicted_tuples = self.find_pair(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
            elif self.args.task == 'triplet':
                predicted_tuples = self.find_triplet(self.predictions[i], predicted_aspect_spans, predicted_opinion_spans, self.tokens_ranges[i])
                if not predicted_tuples:
                  if len(predicted_aspect_spans) > 0:
                    #for j in range(len(predicted_aspect_spans)):
                      #aspect_no_tuple.append([predicted_aspect_spans[j][0],predicted_aspect_spans[j][1]])
                    for k in range(len(predicted_aspect_spans)):
                      aspect_terms = [list_sentences[aspect_index] for aspect_index in predicted_aspect_spans[k]]
                      if len(np.unique(aspect_terms))== 1:
                        aspect_terms = aspect_terms[0]
                      aspect_no_tuple.append(aspect_terms)   
                  #print("NO Triplet Extracted")
                  print("Input Sentence: ",raw_sentences)
                  #print(aspect_no_tuple)
                  triplets_sentences_no_opinion = [{'Aspect_spans':no_tuples,'Opinion_spans': "Empty_opinion",'Sentiment':"Empty_sentiment"} for no_tuples in aspect_no_tuple]
                  result1 = [{'Input Sentence':raw_sentences,'triplets':triplets_sentences_no_opinion}]
                elif len(predicted_tuples)>0:
                  aspect_indices=[]
                  opinion_indices=[]
                  for j in range(len(predicted_tuples)):
                    aspect_indices.append([predicted_tuples[j][0],predicted_tuples[j][1]])
                    opinion_indices.append([predicted_tuples[j][2],predicted_tuples[j][3]])
                    sent.append([predicted_tuples[j][4]])
                  for k in range(len(aspect_indices)):
                    aspect_terms = [list_sentences[aspect_index] for aspect_index in aspect_indices[k]]
                    aspect_tags.append(aspect_terms)
                  for k in range(len(opinion_indices)):
                    opinion_terms = [list_sentences[opinion_index] for opinion_index in opinion_indices[k]]
                    opinion_tags.append(opinion_terms)
                for j in range(len(aspect_tags)):
                  if(len(np.unique(aspect_tags[j]))== 1):
                    aspect_tags[j] = [aspect_tags[j][0]]
                for k in range(len(opinion_tags)):
                  if(len(np.unique(opinion_tags[k]))== 1):
                    opinion_tags[k] = [opinion_tags[k][0]]
                  triplets_sentences = [{'Aspect-Span': aspect_terms, 'Opinion-Span': opinion_terms, 'Sentiment': sentiment} for aspect_terms,opinion_terms,sentiment in zip(aspect_tags,opinion_tags,sent)]
                print("Input Sentence: ",raw_sentences)
                result = [{'Input Sentence':raw_sentences,'triplets':triplets_sentences}]
        return   result if len(triplets_sentences_no_opinion) < 1 else  result1 + result 







            