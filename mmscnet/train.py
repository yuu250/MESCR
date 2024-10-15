import argparse
import time
from math import cos, pi
import torch.nn.functional as F

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from utlis.model import SCNet
import yaml
from sklearn.metrics import classification_report
from utlis.rrnet import  MMSCNet
from utlis.datasets import *
from utlis.metric import *
from timm import create_model
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def adjust_learning_rate(optimizer, current_epoch, max_epoch,hy_dict,lr0=0.0, lrf=0.01, warmup_epochs=0.0, lr_schedule='None'):
    if lr_schedule=='coslr':

        if current_epoch < warmup_epochs:
            lr = lr0 * current_epoch / warmup_epochs
        else:
            lr = lrf + (lr0 - lrf) * (
                    1 + cos(pi * (current_epoch - warmup_epochs) / (max_epoch - warmup_epochs))) / 2

    elif lr_schedule=='stepLR':
        lr = lr0 * (hy_dict['gamma'] ** (current_epoch // hy_dict['step']))

    else:
        lr=lr0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def build_optimizer(model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    if name == 'auto':
        nc = getattr(model, 'nc', 10)  # number of classes
        lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
        name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f'{module_name}.{param_name}' if module_name else param_name
            if 'bias' in fullname:  # bias (no decay)
                g[2].append(param)
            elif isinstance(module, bn):  # weight (no decay)
                g[1].append(param)
            else:  # weight (with decay)
                g[0].append(param)

    if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
        optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(
            f"Optimizer '{name}' not found in list of available optimizers "
            f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
            'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer


def config_model(model, opt, device):
    if opt.pretrained:
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in torch.load(opt.weights).items() if
                           np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if device.type == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    return model

def get_r_loss(y_true, y_pred, y_sigma):
    return (torch.log(y_sigma) / 2 + (y_true - y_pred) ** 2 / (2 * y_sigma)).mean() + 5
def fit_one_epoch(model, train_loader, val_loader, loss_func, current_epoch, max_epoch, opt, names, device):
    global best_acc
    global best_mse
    start_time = time.time()

    # train
    model.train()
    train_pred, train_ture = [], []
    train_loss = 0
    train_mse = 0
    with tqdm(total=len(train_loader.batch_sampler), desc=f'Epoch {current_epoch + 1}/{max_epoch}', postfix=dict,
              mininterval=0.3) as pbar:

        for iteration, batch in enumerate(train_loader):
            #datas, labels = batch[0].to(device), batch[1].to(device)
            gri,urz,pm,px,labels, r=batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device).float()
            outputs = model(gri,urz,pm,px)
            #print(type(outputs[1].squeeze()),type(r))
            loss = loss_func(outputs[0], labels)
            #loss_r = F.mse_loss(outputs[1].squeeze(),r)
            loss_r = get_r_loss(r, outputs[1].squeeze(), outputs[2].squeeze())
            train_loss += loss
            train_loss += loss_r
            train_mse += F.mse_loss(outputs[1].squeeze(), r)
            optimizer.zero_grad()
            (loss+loss_r).backward()
            optimizer.step()

            train_pred += outputs[0].argmax(dim=1).cpu()
            train_ture += labels.cpu()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'train_loss': train_loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'step/s': waste_time})
            pbar.update(1)
            start_time = time.time()

    # Save metrics
    train_loss = train_loss.item() / (iteration + 1)
    train_mse = train_mse.item()/ (iteration + 1)

    train_report = classification_report(train_pred, train_ture, target_names=names, output_dict=True)
    save_metric(f'{opt.save_path}/train', train_report, train_loss,train_mse, names)
    print_result(train_report)
    print(f"train_loss :{train_loss}, train_mse：{train_mse}")

    # Verify
    model.eval()
    val_pred, val_ture = [], []
    val_loss = 0
    val_mse = 0
    with tqdm(total=len(val_loader.batch_sampler), desc=f'Epoch {current_epoch + 1}/{max_epoch}', postfix=dict,
              mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            #datas, labels = batch[0].to(device), batch[1].to(device)
            gri,urz,pm,px,labels,r=batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device).float()
            with torch.no_grad():
                #datas = datas
                r = r.cuda()
                labels = labels.cuda()
                #optimizer.zero_grad()
                outputs = model(gri,urz,pm,px)
                loss = loss_func(outputs[0], labels)
                #loss_r = F.mse_loss(outputs[1].squeeze(), r)
                loss_r = get_r_loss(r,outputs[1].squeeze(),outputs[2].squeeze())
                val_loss += loss
                val_loss += loss_r
                val_mse += F.mse_loss(outputs[1].squeeze(), r)
                val_pred += outputs[0].argmax(dim=1).cpu()
                val_ture += labels.cpu()
            pbar.set_postfix(**{'val_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)

    # Calculate validation metrics and save
    val_loss = val_loss.item() / (iteration + 1)
    val_mse = val_mse.item() / (iteration + 1)
    val_report = classification_report(val_pred, val_ture, target_names=names, output_dict=True)
    save_metric(f'{opt.save_path}/val', val_report, val_loss,val_mse, names)
    print_result(val_report)
    print(f"val loss：{val_loss}, val mse:{val_mse}")

    # Save optimal model
    if (val_report['accuracy'] > best_acc) & (val_mse<best_mse):
        best_acc = val_report['accuracy']
        best_mse = val_mse
        torch.save(model.state_dict(), f'{opt.save_path}/best.pth')

    torch.save(model.state_dict(), f'{opt.save_path}/last.pth')

def train(model, loss_func, train_loader, val_loader, opt, hyp_dict, device):
    for epoch in range(0, hyp_dict['epochs']):
        # Adjust learning rate
        adjust_learning_rate(optimizer=optimizer,
                             current_epoch=epoch + 1,
                             max_epoch=hyp_dict['epochs'],
                             lr0=hyp_dict['lr0'],
                             lrf=hyp_dict['lrf'],
                             warmup_epochs=hyp_dict['warmup_epochs'],
                             lr_schedule=hyp_dict['lr_schedule'],
                             hy_dict=hyp_dict)
        fit_one_epoch(model, train_loader, val_loader, loss_func, epoch, hyp_dict['epochs'], opt, names, device)

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class MultiClassFocalLossWithAlpha(nn.Module): #  label_map = {'O': 0, 'A': 1, 'K': 2, 'G': 3, 'M': 4, 'B': 5, 'F': 6}
    def __init__(self, alpha=[1/477, 1/3320,1/4226,1/2814,1/1106,1/363,1/5709], gamma=1, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha.to(target.device)  # 移动到 target 相同的设备
        alpha = alpha[target]
        #alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

if __name__ == '__main__':
    # Parameter settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, default='result/flame/new/imgonly/', help='result storage address')
    parser.add_argument('--data', type=str, default='config/dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='config/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--model', type=str, default='None', help='model name')
    parser.add_argument('--combination', type=str, default='gri+urz', help='bound combination')
    parser.add_argument('--pretrained', type=bool, default=False, help='whether start pre training?')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--single-cls', action='store_true', default=False,
                        help='train as single-class dataset')

    opt = parser.parse_args()

    # Dataset information loading
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    nc, names = (1, ['item']) if opt.single_cls else (int(data_dict['nc']), data_dict['names'])  # number classes, names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Loading hyperparameter information
    with open(opt.hyp, encoding='utf-8') as f:
        hyp_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict


    def create_path_if_not_exists(path):
        if not os.path.exists(path):
            os.makedirs(path)
    # 确保路径存在
    create_path_if_not_exists(f'{opt.save_path}train/')

    with open(f'{opt.save_path}train/dataset.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(data_dict, file, allow_unicode=True)

    with open(f'{opt.save_path}train/hyp.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(hyp_dict, file, allow_unicode=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #set seed
    seed=hyp_dict['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Get data
    train_loader, val_loader = get_train_dataloader(data_dict, hyp_dict,opt.combination)

    if opt.model!='None':
        hyp_dict['model']=opt.model
    # Build model
    if hyp_dict['model']=='SCNet':
        model = SCNet(nc=nc)
    elif hyp_dict['model']=='MMSCNet':
        model = MMSCNet(nc=nc)

    model = config_model(model, opt, device)

    # loss function[1/477, 1/3320,1/4226,1/2814,1/1106,1/363,1/5709],
    #loss_func = nn.CrossEntropyLoss(weight=torch.tensor([4.6, 1.3, 1.1, 1.3, 1.5, 3.0, 1.0]).to(device))#[ 'O', 'A', 'K','G', 'M','B','F']
    #loss_func = nn.CrossEntropyLoss(weight=torch.tensor([3.8, 2.6, 2.0, 3.4, 4.0, 1.4, 1.0]).to(device))
    loss_func = nn.CrossEntropyLoss()
    #loss_func=LabelSmoothingCrossEntropy()
    #loss_func = MultiClassFocalLossWithAlpha()
    # optimizer
    optimizer = build_optimizer(model,
                                name=hyp_dict['optimizer'],
                                lr=hyp_dict['lr0'],
                                momentum=hyp_dict['momentum'])

    best_acc = 0.0
    best_mse = 999
    # train
    train(model, loss_func, train_loader, val_loader, opt, hyp_dict, device)

