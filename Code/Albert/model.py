import torch
import torch.nn

from transformers import AlbertConfig,AlbertModel,AlbertTokenizer


class MultiInferalBert(torch.nn.Module):
    def __init__(self, args):
        super(MultiInferalBert, self).__init__()

        self.args = args
        #self.config = AlbertConfig.from_pretrained( args.albert_model_path)
        self.albert = AlbertModel.from_pretrained(args.albert_model_path)
        self.tokenizer = AlbertTokenizer.from_pretrained(args.albert_tokenizer_path)

        self.cls_linear = torch.nn.Linear(args.albert_feature_dim*2, args.class_num)
        self.feature_linear = torch.nn.Linear(args.albert_feature_dim*2 + args.class_num*3, args.albert_feature_dim*2)
        self.dropout_output = torch.nn.Dropout(0.1)

    def multi_hops(self, features, mask, k):
        '''generate mask'''
        max_length = features.shape[1]
        mask = mask[:, :max_length]
        mask_a = mask.unsqueeze(1).expand([-1, max_length, -1])
        mask_b = mask.unsqueeze(2).expand([-1, -1, max_length])
        mask = mask_a * mask_b
        mask = torch.triu(mask).unsqueeze(3).expand([-1, -1, -1, self.args.class_num])

        '''save all logits'''
        logits_list = []
        logits = self.cls_linear(features)
        logits_list.append(logits)

        for i in range(k):
            #probs = torch.softmax(logits, dim=3)
            probs = logits
            logits = probs * mask

            logits_a = torch.max(logits, dim=1)[0]
            logits_b = torch.max(logits, dim=2)[0]
            logits = torch.cat([logits_a.unsqueeze(3), logits_b.unsqueeze(3)], dim=3)
            logits = torch.max(logits, dim=3)[0]

            logits = logits.unsqueeze(2).expand([-1,-1, max_length, -1])
            logits_T = logits.transpose(1, 2)
            logits = torch.cat([logits, logits_T], dim=3)

            new_features = torch.cat([features, logits, probs], dim=3)
            features = self.feature_linear(new_features)
            logits = self.cls_linear(features)
            logits_list.append(logits)
        return logits_list

    def forward(self, tokens, masks):
        albert_feature,_ = self.albert(tokens, masks)
        albert_feature = self.dropout_output(albert_feature)

        albert_feature = albert_feature.unsqueeze(2).expand([-1, -1, self.args.max_sequence_len, -1])
        albert_feature_T = albert_feature.transpose(1, 2)
        features = torch.cat([albert_feature, albert_feature_T], dim=3)
        logits = self.multi_hops(features, masks, self.args.nhops)

        return logits[-1]