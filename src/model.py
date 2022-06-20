import torch
from torch.cuda.amp import autocast as autocast

from collections import Counter
from sklearn.metrics import *
from tqdm import tqdm
import json
import copy

from dataset import SSLdataset, get_dataLoader
from TYY_stodepth_lineardecay import buildModel
from utils import *

class Model:
    def __init__(self, buildModel, numClasses, num_labels=250, T=0.5, p_cutoff=0.95, lambda_u=1.0, num_eval_epoch=5, max_epoch=200, batch_size=12, isFlex=False):
        """
        buildModel: function to build a model (resnet50)
        ema: momentum of exponential moving average for eval_model
        p_cutoff: confidence cutoff parameters for loss masking
        lambda_u: ratio of unsupervised loss to supervised loss
        num_eval_iter: freqeuncy of iteration
        """
        super(Model, self).__init__()

        if  batch_size > num_labels:
            batch_size = num_labels

        self.model = buildModel(numClasses, isFlex)
        self.numClasses = numClasses
        self.p_cutoff = p_cutoff
        self.lambda_u = lambda_u
        self.num_eval_epoch = num_eval_epoch
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.T = T

        train_dset = SSLdataset()
        self.lb_dset, self.ulb_dset = train_dset.get_ssl_dset(self.num_labels)
        _eval_dset = SSLdataset(train=False)
        self.eval_dset = _eval_dset.get_dset()

        loader_dict = {}
        loader_dict["train_lb"] = get_dataLoader(self.lb_dset, self.batch_size, True, 1, False)
        loader_dict["train_ulb"] = get_dataLoader(self.ulb_dset, self.batch_size, True, 1, False)
        loader_dict["eval"] = get_dataLoader(self.eval_dset, 1024, True, 1, False, False)
        self.loader_dict = loader_dict

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict

    def set_dset(self, dset):
        self.ulb_dset = dset

    def train(self):
        use_cuda = torch.cuda.is_available()
        device =  torch.device("cuda:0" if use_cuda else "cpu")

        lr = 0.0005
        momentum = 0.9
        weight_decay = 5e-4

        # lr = 0.001
        # weight_decay = 5e-4

        self.model = self.model.to(device)
        paras = self.model.parameters()
        optimizer = torch.optim.SGD(paras, lr=lr, momentum=momentum, weight_decay=weight_decay)
        # optimizer = torch.optim.Adam(paras, lr=lr, weight_decay=weight_decay)

        # p(y) based on labeled examples seen during training
        p_model = None
        dist_filename = "./data/labels_"+str(self.num_labels)+".json"
        with open(dist_filename, 'r') as f:
                p_target = json.loads(f.read())
                p_target = torch.tensor(p_target['distribution'])
                p_target = p_target.to(device)


        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * (-1)
        selected_label = selected_label.to(device)

        classwise_acc = torch.zeros((self.numClasses,)).to(device)

        best_eval_acc, best_epoch = 0.0, 0
        steps = 0
        acc_list = []
        for epoch in range(self.max_epoch):
            running_loss = 0.0
            devider = 0.0
            for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s1, x_ulb_s2) in zip(self.loader_dict["train_lb"],
                                                                    self.loader_dict["train_ulb"]):
                num_lb = x_lb.shape[0]
                num_ulb = x_ulb_w.shape[0]
                assert num_ulb == x_ulb_s1.shape[0]

                x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = x_lb.to(device), x_ulb_w.to(device), x_ulb_s1.to(device), x_ulb_s2.to(device)
                x_ulb_idx = x_ulb_idx.to(device)
                y_lb = y_lb.to(device)

                pseudo_counter = Counter(selected_label.tolist())
                if max(pseudo_counter.values()) < len(self.ulb_dset):
                    wo_negative_one = copy.deepcopy(pseudo_counter)
                    if -1 in wo_negative_one.keys():
                        wo_negative_one.pop(-1)
                    for i in range(self.numClasses):
                        classwise_acc[i] = pseudo_counter[i] / max(wo_negative_one.values())
                
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2))

                # model
                optimizer.zero_grad()
                logits = self.model(inputs)
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s1, logits_x_ulb_s2 = logits[num_lb:].chunk(3)
                logits_x_ulb_s = (logits_x_ulb_s1+logits_x_ulb_s2)/2 # for ours
                # logits_x_ulb_s = logits_x_ulb_s1 # for flex
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                unsup_loss, mask, select, pseudo_lb, p_model = consistency_loss(logits_x_ulb_s,
                                                                    logits_x_ulb_w,
                                                                    classwise_acc,
                                                                    p_target,
                                                                    p_model,
                                                                    'ce', self.T, 
                                                                    p_cutoff=self.p_cutoff,
                                                                    use_hard_labels=True)

                if x_ulb_idx[select == 1].nelement() != 0:
                    selected_label[x_ulb_idx[select == 1]] = pseudo_lb[select == 1]

                total_loss = sup_loss + self.lambda_u * unsup_loss
                running_loss += total_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=paras, max_norm=10)
                optimizer.step()
            
                devider += 1.0
                steps += 1

            epoch_loss = running_loss / devider
            print(f"epoch: [{epoch+1}/{self.max_epoch}], epoch loss: {epoch_loss:.4f}")

            if (epoch+1) % self.num_eval_epoch == 0:
                eval_dict = self.evaluate()
                running_acc = eval_dict["eval/top-1-acc"]
                acc_list.append(running_acc)

                print(f'>>>>>>>>>>>> epoch: [{epoch+1}/{self.max_epoch}], step: {steps}, eval acc:{running_acc:.4f}')

                if running_acc > best_eval_acc:
                    part_save_path = "./data/models/part_train_"+str(self.num_labels)+f"/{epoch+1}_{running_acc:.4f}.pth"
                    torch.save(self.model.state_dict(), part_save_path)
                    best_eval_acc=running_acc
                    best_epoch = epoch
                
            if (epoch+1) % 20 == 0:
                lr /= 3
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

        save_path = f"./data/models/model_{self.num_labels}.pth"
        torch.save(self.model.state_dict(), save_path)
        print(f"Model is saved at {save_path}")
        print(f"Best acc is {best_eval_acc:.4f} at epoch {best_epoch}")
        print("acc list:",acc_list)

    @torch.no_grad()
    def evaluate(self, model_path=None):
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        use_cuda = torch.cuda.is_available()
        device =  torch.device("cuda:0" if use_cuda else "cpu")

        self.model = self.model.to(device)

        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []

        for _, x, y in self.loader_dict["eval"]:
            x, y = x.to(device), y.to(device)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top3 = top_k_accuracy_score(y_true, y_logits, k=3)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')

        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-3-acc': top3,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    model = Model(buildModel, numClasses=10, isFlex=False)
    model.train()
    # result=model.evaluate("data/models/best_ours.pth")
    # print(result)