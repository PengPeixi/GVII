"""
This code is developed based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""

import torch
import torch.nn as nn
from attention import BiAttention, StackedAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from utils import tfidf_loading
from simple_cnn import SimpleCNN, SimpleCNN32
from learner import MAML
from auto_encoder import Auto_Encoder_Model

from g_mlp_pytorch import gMLPVision
from multi_head_attention import MultiHeadedAttention
#from vit_model import vit_base_patch16_224_in21k as create_model
#from models import get_model

#from torchvision.transforms import Resize


#import dgl
#import networkx as nx
#from graph_transformer_layer import GraphTransformerLayer_no_edge
#import numpy as np

# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb, ae_v_emb):
    #def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,ae_v_emb, img_only_model):
        super(BAN_Model, self).__init__()
        self.args = args
        self.dataset = dataset
        self.op = args.op
        self.glimpse = args.gamma
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        if counter is not None:  # if do not use counter
            self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()
        if args.maml:
            # init multiple maml models
            if len(self.args.maml_nums) > 1:
                self.maml_v_emb = nn.ModuleList(model for model in maml_v_emb)
            else:
                self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            self.convert = nn.Linear(16384, args.feat_dim)

        #self.fact_graph_transformer = GraphTransformerLayer_no_edge(1024, 1024, 8)

        if 'RAD' in self.args.VQA_dir:
            self.rad_v_mlp = gMLPVision(
                image_size=84,
                patch_size=12,
                num_classes=dataset.num_ans_candidates,
                dim=512,
                depth=6,
                channels = 1
            )

        else:
            #self.v_mlp = img_only_model
            #self.v_mlp = gMLPVision(
            #    image_size=84,
            #    patch_size=12,
            #    num_classes=dataset.num_ans_candidates,
            #    #num_classes=1024,
            #    dim=512,
            #    depth=6
            #)

            # self.v_mlp = img_only_model
            self.v_mlp = gMLPVision(
                image_size=84,
                patch_size=14,
                num_classes=dataset.num_ans_candidates,
                #    #num_classes=1024,
                dim=768,
                depth=6
            )

        self.fact_mlp = nn.Sequential(
            nn.Linear(1024*12, 512*12),
            nn.ReLU(inplace=True),
            nn.Linear(512*12, 256*12),
            nn.ReLU(inplace=True),
            nn.Linear(256*12, dataset.num_ans_candidates)
        )

        self.vit_mlp = nn.Sequential(
            nn.Conv2d(1,1,(4,1),stride=(2,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1,1,(3,1),stride=(2,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1,1,(2,1)),
            nn.ReLU(inplace=True),
            #nn.Linear(1024, 2048),
            #nn.ReLU(inplace=True),
            #nn.Linear(2048, dataset.num_ans_candidates)
        )

        self.vit_mlp_2 = nn.Sequential(
            nn.Linear(dataset.num_ans_candidates, 1024*2),
            nn.ReLU(inplace=True),
            nn.Linear(1024*2, 1024),
        )

        self.vit_mlp_3 = nn.Sequential(
            nn.Linear(1024, 1024*2),
            nn.ReLU(inplace=True),
            nn.Linear(1024*2, dataset.num_ans_candidates),
        )

        self.cross_att = MultiHeadedAttention(8, 1024)
        self.self_att_v_before = MultiHeadedAttention(8, 1024)
        self.self_att_l_before = MultiHeadedAttention(8, 1024)
        self.self_att_v_after = MultiHeadedAttention(8, 1024)
        #self.norm_net = nn.Softmax(dim=1)
        #self.norm_net_1 = nn.Softmax(dim=1)

        self.constant_k = nn.Parameter(torch.tensor(0.0))
        self.constant_v = nn.Parameter(torch.tensor(0.0))
        #self.constant_add = nn.Parameter(torch.tensor(0.0))
        #self.constant_add_1 = nn.Parameter(torch.tensor(0.0))

        #self.torch_resize = Resize([224, 224])

    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        if self.args.maml: # get maml feature
            # compute multiple maml embeddings and concatenate them
            tsne_tempt_maml_feat_lst = []
            if len(self.args.maml_nums) > 1:
                maml_v_emb = self.maml_v_emb[0](v[0]).unsqueeze(1)
                #print(self.maml_v_emb[0].net.feature.shape)
                #tsne_tempt_maml_feat_lst.append(self.maml_v_emb[0].net.feature)
                for j in range(1, len(self.maml_v_emb)):
                    #tsne_tempt_maml_feat_lst.append(self.maml_v_emb[j].net.feature)
                    maml_v_emb_temp = self.maml_v_emb[j](v[0]).unsqueeze(1)
                    maml_v_emb = torch.cat((maml_v_emb, maml_v_emb_temp), 2)
            else:
                maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        #tsne_tempt_maml_feat = maml_v_emb.clone()
        if self.args.autoencoder: # get dae feature
            encoder = self.ae_v_emb.forward_pass(v[1])
            decoder = self.ae_v_emb.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder: # concatenate maml feature with dae feature
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)

        # get lextual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]

        #device = torch.device("cuda", 0)
        #fact_graphs = []
        #for i in range(q_emb.shape[0]):
        #    graph = dgl.DGLGraph().to(device)
        #    graph.add_nodes(q_emb[i].shape[0])
        #    graph.add_edges(list(range(11)), list(range(1,12)))
        #    graph.ndata['h'] = q_emb[i]
        #    graph_ndata_h = self.fact_graph_transformer(graph, graph.ndata['h'])
        #    if i == 0:
        #        fact_embs = graph_ndata_h.unsqueeze(0)
        #    else:
        #        fact_embs = torch.cat((fact_embs, graph_ndata_h.unsqueeze(0)), 0)
        #q_emb = fact_embs

        # Attention
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb) # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:,g,:,:]) # b x l x h
            atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        if 'RAD' in self.args.VQA_dir:
            v_pred = self.rad_v_mlp(v[0])
        else:
            #v_pred = self.v_mlp(v[0])
            #v_vit_feat = self.v_mlp(self.torch_resize(v[0]))
            #v_pred = self.v_mlp(self.torch_resize(v[0]))
            v_vit_feat = self.v_mlp(v[0])

        #tsne_tempt_gmlp_feat = v_vit_feat.clone()
        #v_pred = self.vit_mlp(v_pred)
        #q_emb_reduce = self.q_emb.forward_all(w_emb)
        q_emb_reduce = self.vit_mlp(q_emb.unsqueeze(1))
        v_vit_feat_before = self.vit_mlp_2(v_vit_feat)
        v_vit_feat_after = self.self_att_v_before(v_vit_feat_before,v_vit_feat_before,v_vit_feat_before).squeeze(1)
        q_emb_feat = self.self_att_l_before(q_emb_reduce,q_emb_reduce,q_emb_reduce).squeeze(1)

        v_pred_after = self.cross_att(q_emb_feat,v_vit_feat_after,v_vit_feat_after).squeeze(1)
        v_pred_after = self.self_att_v_after(v_pred_after,v_pred_after,v_pred_after).squeeze(1)
        #v_pred = self.norm_net(self.vit_mlp_2(self.constant_add * v_pred_after + self.constant_add_1 * v_vit_feat))
        v_pred = 0.01 * self.vit_mlp_3(v_pred_after) + v_vit_feat
        #v_pred = self.constant_add * v_pred_after + self.vit_mlp_2(v_vit_feat).squeeze(1)
        #v_pred = self.constant_add * v_pred_after + v_vit_feat

        fact_mlp_input = q_emb.reshape(q_emb.shape[0], -1)
        fact_pred = self.fact_mlp(fact_mlp_input)

        preds = self.classifier(q_emb.sum(1))
        eps = 1e-12
        pred_co_all = torch.log(torch.sigmoid(preds + self.constant_k * fact_pred + self.constant_v * v_pred) + eps)

        return pred_co_all, decoder, fact_pred, v_pred
            #, tsne_tempt_gmlp_feat, tsne_tempt_maml_feat

        #if self.args.autoencoder:
        #        return q_emb.sum(1), decoder, fact_pred

        #return q_emb.sum(1)

    #def classify(self, input_feats):
        #return self.classifier(input_feats)

# Create SAN model
class SAN_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb):
        super(SAN_Model, self).__init__()
        self.args = args
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier
        if args.maml:
            # init multiple maml models
            if len(self.args.maml_nums) > 1:
                self.maml_v_emb = nn.ModuleList(model for model in maml_v_emb)
            else:
                self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            self.convert = nn.Linear(16384, args.feat_dim)
    def forward(self, v, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # get visual feature
        if self.args.maml: # get maml feature
            # compute multiple maml embeddings and concatenate them
            if len(self.args.maml_nums) > 1:
                maml_v_emb = self.maml_v_emb[0](v[0]).unsqueeze(1)
                for j in range(1, len(self.maml_v_emb)):
                    maml_v_emb_temp = self.maml_v_emb[j](v[0]).unsqueeze(1)
                    maml_v_emb = torch.cat((maml_v_emb, maml_v_emb_temp), 2)
            else:
                maml_v_emb = self.maml_v_emb(v[0]).unsqueeze(1)
            v_emb = maml_v_emb
        if self.args.autoencoder: # get dae feature
            encoder = self.ae_v_emb.forward_pass(v[1])
            decoder = self.ae_v_emb.reconstruct_pass(encoder)
            ae_v_emb = encoder.view(encoder.shape[0], -1)
            ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
            v_emb = ae_v_emb
        if self.args.maml and self.args.autoencoder: # concatenate maml feature with dae feature
            v_emb = torch.cat((maml_v_emb, ae_v_emb), 2)
        # get textual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim], return final hidden state
        # Attention
        att = self.v_att(v_emb, q_emb)
        if self.args.autoencoder:
            return att, decoder
        return att

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Build BAN model
def build_BAN(dataset, args, priotize_using_counter=False):

    #device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    #img_only_model = create_model(num_classes=1024, has_logits=True).to(device)
    #img_only_model = create_model(num_classes=dataset.num_ans_candidates, has_logits=True).to(device)
    #weights_dict = torch.load(args.weights, map_location=device)
    #del_keys = ['head.weight', 'head.bias'] if img_only_model.has_logits \
    #    else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    #for k in del_keys:
    #    del weights_dict[k]
    #print(img_only_model.load_state_dict(weights_dict, strict=False))
    #img_only_model = volo_d1(img_size=84,num_classes=3974).to(device)
    #img_only_model = get_model('PoolFormer', 'S24', pretrained='pretrain_model_weights/poolformer_s24.pth.tar').to(device)
    #img_only_model = get_model('PoolFormer', 'S24', pretrained=None, num_classes=dataset.num_ans_candidates, image_size=84).to(device)

    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0,  args.rnn)
    v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
    # build and load pre-trained MAML model(s)
    if args.maml:
        # load multiple pre-trained maml models(s)
        if len(args.maml_nums) > 1:
            maml_v_emb = []
            for model_t in args.maml_nums:
                weight_path = args.VQA_dir + '/maml/' + 't%s_'%(model_t) + args.maml_model_path
                print('load initial weights MAML from: %s' % (weight_path))
                # maml_v_emb = SimpleCNN32(weight_path, args.eps_cnn, args.momentum_cnn)
                maml_v_emb_temp = MAML(args.VQA_dir)
                maml_v_emb_temp.load_state_dict(torch.load(weight_path))
                maml_v_emb.append(maml_v_emb_temp)
        else:
            weight_path = args.VQA_dir + '/maml/' + 't%s_' % (args.maml_nums[0]) + args.maml_model_path
            print('load initial weights MAML from: %s' % (weight_path))
            # maml_v_emb = SimpleCNN32(weight_path, args.eps_cnn, args.momentum_cnn)
            maml_v_emb = MAML(args.VQA_dir)
            maml_v_emb.load_state_dict(torch.load(weight_path))
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.VQA_dir + '/' + args.ae_model_path
        print('load initial weights DAE from: %s'%(weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # Optional module: counter for BAN
    use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter
    if use_counter or priotize_using_counter:
        objects = 10  # minimum number of boxes
    if use_counter or priotize_using_counter:
        counter = Counter(objects)
    else:
        counter = None
    # init BAN residual network
    b_net = []
    q_prj = []
    c_prj = []
    for i in range(args.gamma):
        b_net.append(BCNet(dataset.v_dim, args.num_hid, args.num_hid, None, k=1))
        q_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
        if use_counter or priotize_using_counter:
            c_prj.append(FCNet([objects + 1, args.num_hid], 'ReLU', .0))
    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args)
    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                         ae_v_emb)
#        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
#                         ae_v_emb, img_only_model)
    elif args.maml:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb,
                         None)
    elif args.autoencoder:
        return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None,
                         ae_v_emb)
    return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, None, None)

# Build SAN model
def build_SAN(dataset, args):
    # init word embedding module, question embedding module, and Attention network
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0, args.rnn)
    v_att = StackedAttention(args.num_stacks, dataset.v_dim, args.num_hid, args.num_hid, dataset.num_ans_candidates,
                             args.dropout)
    # build and load pre-trained MAML model(s)
    if args.maml:
        # load multiple pre-trained maml models(s)
        if len(args.maml_nums) > 1:
            maml_v_emb = []
            for model_t in args.maml_nums:
                weight_path = args.VQA_dir + '/maml/' + 't%s_'%(model_t) + args.maml_model_path
                print('load initial weights MAML from: %s' % (weight_path))
                # maml_v_emb = SimpleCNN32(weight_path, args.eps_cnn, args.momentum_cnn)
                maml_v_emb_temp = MAML(args.VQA_dir)
                maml_v_emb_temp.load_state_dict(torch.load(weight_path))
                maml_v_emb.append(maml_v_emb_temp)
        else:
            weight_path = args.VQA_dir + '/maml/' + 't%s_' % (args.maml_nums[0]) + args.maml_model_path
            print('load initial weights MAML from: %s' % (weight_path))
            maml_v_emb = MAML(args.VQA_dir)
            maml_v_emb.load_state_dict(torch.load(weight_path))
    # build and load pre-trained Auto-encoder model
    if args.autoencoder:
        ae_v_emb = Auto_Encoder_Model()
        weight_path = args.VQA_dir + '/' + args.ae_model_path
        print('load initial weights DAE from: %s'%(weight_path))
        ae_v_emb.load_state_dict(torch.load(weight_path))
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)
    # init classifier
    classifier = SimpleClassifier(
        args.num_hid, 2 * args.num_hid, dataset.num_ans_candidates, args)

    # contruct VQA model and return
    if args.maml and args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, ae_v_emb)
    elif args.maml:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, maml_v_emb, None)
    elif args.autoencoder:
        return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, ae_v_emb)
    return SAN_Model(w_emb, q_emb, v_att, classifier, args, None, None)