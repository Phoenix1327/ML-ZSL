
###############################
# cls_predict: N*C, e.g., N*81
pos_loss = torch.mul(pos_weight_, torch.mul(label_, torch.log(1e-5 + cls_predict)))
pos_loss = torch.sum(pos_loss, dim=0, keepdim=True)
neg_loss = torch.mul(neg_weight_, torch.mul(1-label_, torch.log(1e-5 + 1 - cls_predict)))
neg_loss = torch.sum(neg_loss, dim=0, keepdim=True)

num_pos = torch.sum(pos_weight_, dim=0, keepdim=True)
numPos = torch.sum(num_pos>0)
num_pos[num_pos==0] = 1
num_neg = torch.sum(neg_weight_, dim=0, keepdim=True)
numNeg = torch.sum(num_neg>0)
num_neg[num_neg==0] = 1

pos_loss = torch.div(pos_loss, num_pos)
loss_pos_bce = -pos_loss.mean()
neg_loss = torch.div(neg_loss, num_neg)
loss_neg_bce = -neg_loss.mean()

###############################
# cls_predict: N*C, e.g., N*81
loss_pos_bce = F.binary_cross_entropy(cls_predict, label_, pos_weight_, size_average=False)
loss_pos_bce = loss_pos_bce / torch.sum(pos_weight_)
loss_neg_bce = F.binary_cross_entropy(cls_predict, label_, neg_weight_, size_average=False)
loss_neg_bce = loss_neg_bce / torch.sum(neg_weight_)
loss_img_bce = torch.mean(loss_pos_bce + loss_neg_bce)
###############################
#pred: N*C, prediction
#gt: N*C, ground truth
for i in range(num_classes):
    single_pred = pred[:, i]
    single_gt = gt[:, i]
    single_accuracy = accuracy_score(single_gt, (single_pred > 0.5).astype(float))
    single_roc_score = roc_auc_score(single_gt, single_pred)
    single_ap_score = average_precision_score(single_gt, single_pred)

###############################
# vae train
def _vae_train(self, epoch, results):
    print("Optimization VAE: Epoch----{0}".format(epoch))
    self.VAE.train()
    graph_emb = self._gae_eval()

    for i, (image_, label_, weight_) in enumerate(self.data_loader):
        image_ = self._get_variable(image_)
        batchsize = image_.shape[0]
        label_ = self._get_variable(label_)
        weight_ = self._get_variable(weight_)
        img_emb = self._extract_features(image_)
        self.VAE.zero_grad()

        recon_emb, vae_num, vae_std = self.VAE(img_emb)
        # Reconstruction + KL divergence losses
        vae_mse = F.mse_loss(recon_emb, img_emb, size_average=True)
        vae_kld = -(0.5 / batchsize) * torch.mean(torch.sum(1 + 2*torch.log(vae_std) - vae_mu.pow(2) - vae_std.pow(2), 1))
        loss_vae = vae_mse + vae_kld

        graph_emb = self._gae_eval()
        cls_predict = torch.mm(vae_mu, graph_emb.detach().t())
        loss_img_bce = F.binary_cross_entropy(cls_predict, label_, weight_, size_average=True)

        loss_img = loss_vae + loss_img_bce
        loss_img.backward()
        self.optimizer_vae.step()

        results['vae_mse'].append(vae_mse.cpu().data[0])
        results['vae_kld'].append(vae_kld.cpu().data[0])
        results['loss_vae'].append(loss_vae.cpu().data[0])
        results['loss_img_bce'].append(loss_img_bce.cpu().data[0])

        if i % 10 == 0:
            print("Optimization VAE: Epoch{0}/Iter{1}".format(epoch, i),
                  "VAE loss: {:.5f}".format(loss_vae.cpu().data[0]),
                  "MSE loss: {:.5f}".format(vae_mse.cpu().data[0]),
                  "KLD loss: {:.5f}".format(vae_kld.cpu().data[0]),
                  "CLS loss: {:.5f}".format(loss_img_bce.cpu().data[0]))

    return results
