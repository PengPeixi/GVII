# Create GVII model
class GVII_Model(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args, maml_v_emb, ae_v_emb):
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
        if counter is not None:
            self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()
        if args.maml:
            if len(self.args.maml_nums) > 1:
                self.maml_v_emb = nn.ModuleList(model for model in maml_v_emb)
            else:
                self.maml_v_emb = maml_v_emb
        if args.autoencoder:
            self.ae_v_emb = ae_v_emb
            self.convert = nn.Linear(16384, args.feat_dim)

        self.v_mlp = gMLPVision(
            image_size=84,
            patch_size=14,
            num_classes=dataset.num_ans_candidates,
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

        self.constant_k = nn.Parameter(torch.tensor(0.0))
        self.constant_v = nn.Parameter(torch.tensor(0.0))


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

        # get lextual feature
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]

        # Attention
        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v_emb, q_emb) # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v_emb, q_emb, att[:,g,:,:]) # b x l x h
            atten, _ = logits[:,g,:,:].max(2)
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb

        v_vit_feat = self.v_mlp(v[0])

        q_emb_reduce = self.vit_mlp(q_emb.unsqueeze(1))
        v_vit_feat_before = self.vit_mlp_2(v_vit_feat)
        v_vit_feat_after = self.self_att_v_before(v_vit_feat_before,v_vit_feat_before,v_vit_feat_before).squeeze(1)
        q_emb_feat = self.self_att_l_before(q_emb_reduce,q_emb_reduce,q_emb_reduce).squeeze(1)

        v_pred_after = self.cross_att(q_emb_feat,v_vit_feat_after,v_vit_feat_after).squeeze(1)
        v_pred_after = self.self_att_v_after(v_pred_after,v_pred_after,v_pred_after).squeeze(1)
        v_pred = v_pred_constant * self.vit_mlp_3(v_pred_after) + v_vit_feat

        fact_mlp_input = q_emb.reshape(q_emb.shape[0], -1)
        fact_pred = self.fact_mlp(fact_mlp_input)

        preds = self.classifier(q_emb.sum(1))
        pred_co_all = torch.log(torch.sigmoid(preds + self.constant_k * fact_pred + self.constant_v * v_pred) + eps)

        return pred_co_all, decoder, fact_pred, v_pred


