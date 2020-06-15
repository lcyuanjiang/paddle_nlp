#coding:utf-8
import torch.nn as nn
from model.modeling_bert import BertModel
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import time
from datetime import timedelta
from model.optimization import BertAdam
from tensorboardX import  SummaryWriter
import argparse
from data_process.build_data_bin import *
from config.Config import Config
from torch.utils.data import DataLoader


def get_time_dif(start_time):
    end_time=time.time()
    time_dif=end_time-start_time
    return timedelta(seconds=int(round(time_dif)))

def to_devie(data,device):
    if isinstance(data,(list,tuple)):
        return [to_devie(x,device)for x in data]
    return data.to(device,non_blocking=True)

class Model(nn.Module):
    def __init__(self,config):
        super(Model,self).__init__()
        self.bert=BertModel.from_pretrained(config.bert_pretrain_model)
        num=0
        for param in self.bert.parameters():
            param.requires_grad=True
            num=num+1
        print(num)
        self.fc=nn.Linear(config.hidden_size,config.num_class)

    def forward(self, input_x,mask):
        _,pooled=self.bert(input_x,attention_mask=mask)
        out=self.fc(pooled)
        return out


def train(config,model,train_iter,dev_iter,test_iter):
    model.train()
    if config.use_cuda:
        print("加载模型到GPU.....")
        model.cuda()
    start_time=time.time()
    param_optimizer=list(model.named_parameters())
    no_decay=['bias','LayerNorm.bias','LayerNorm.weigth']
    for n,p in param_optimizer:
        print(n,p)
    paramizer_grouped_parameters=[
        #n中没有任何在no_decay中的参数
        {'params':[ p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
        #n中任何在no_decay中的参数
        {'params':[ p for n,p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay':0.0}
                                ]
    optimizer=BertAdam(paramizer_grouped_parameters,
                       lr=config.learning_rate,
                       warmup=0.05,
                       t_total=len(train_iter)*config.num_epoch
                             )
    total_batch=0
    dev_best_loss=float('inf')
    last_improve=0#最后提升的batch
    flag=False#记录是否提升
    writer=SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M',time.localtime()))
    for epoch in range(config.num_epoch):
        print("Epoch [{}/{}]".format(epoch+1,config.num_epoch))
        for train_data in train_iter:
            train_data_X=train_data[0]
            labels=train_data[1]
            train_mask=train_data[2]
            if config.use_cuda:
                train_data_X=train_data_X.cuda()
                labels=labels.cuda()
                train_mask=train_mask.cuda()
            model.zero_grad()
            output=model(train_data_X,train_mask)
            loss=F.cross_entropy(output,labels)
            loss.backward()
            optimizer.step()

            if total_batch %100==0:
                true=labels.data.cpu()
                predict=torch.max(output.data,1)[1].cpu()
                train_acc=metrics.accuracy_score(true,predict)
                print("训练集loss:", loss)
                print("训练集精度:", train_acc)
                dev_acc,dev_loss=evaluate(config,model,dev_iter)
                print("验证集精度:",dev_acc)
                if dev_acc<dev_best_loss:
                    dev_best_loss=dev_loss
                    torch.save(model.state_dict(),config.save_path)
                    last_improve=total_batch#最后提升的一个batch
                    improve="*"
                else:
                    improve=""
                time_dif=get_time_dif(start_time)
                msg='迭代次数：{0:>6},训练误差：{1:>5.2},训练精度：{2:>6.2%},验证误差：{3:>5.2},验证精度：{4:>6.2%}，花费时间：{5} {6}'
                print(msg.format(total_batch,loss.item(),train_acc,dev_loss,dev_acc,time_dif,improve))
                writer.add_scalar("loss_train",loss,total_batch)
                writer.add_scalar("loss_dev", dev_loss, total_batch)
                writer.add_scalar("acc_train", train_acc, total_batch)
                writer.add_scalar("acc_dev", dev_acc, total_batch)
                model.train()
            total_batch=total_batch+1
            if total_batch-last_improve>config.require_improvement:
                print("{}次误差都没有下降，停止迭代...".format(config.require_improvement))
                flag=True
                break
        if flag:
            break
    writer.close()
    test(config,model,test_iter)

def evaluate(config,model,data_iter,test=False):
    model.eval()
    loss_total=0
    predict_all=np.array([],dtype=int)
    labels_all=np.array([],dtype=int)
    with torch.no_grad():
        for data in data_iter:
            data_X = data[0]
            labels = data[1]
            mask = data[2]
            if config.cuda:
                data_X = data_X.cuda()
                labels = labels.cuda()
                mask = mask.cuda()

            outputs=model(data_X,mask)
            loss=F.cross_entropy(outputs,labels)
            loss_total=loss_total+loss

            labels=labels.data.cpu().numpy()
            predict=torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all=np.append(labels_all,labels)
            predict_all=np.append(predict_all,predict)
    acc=metrics.accuracy_score(labels_all,predict_all)
    loss_total_avg=loss_total/len(data_iter)
    if test:
        report=metrics.classification_report(labels_all,predict_all,target_names=config.mul_class_list,digits=4)
        confusion=metrics.confusion_matrix(labels_all,predict_all)
        return acc,loss_total_avg,report,confusion
    return acc,loss_total_avg

def test(config,model,test_iter):
    print("测试集预测".center(40,'_'))
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time=time.time()
    test_acc,test_loss,test_report,test_confusion=evaluate(config,model,test_iter,test=True)
    msg="测试集损失：{0:>5.2},测试集准确率：{1:>6.2}"
    print(msg.format(test_loss,test_acc))
    print("准确率，召回率，F1...")
    print(test_report)
    print("混淆矩阵...")
    print(test_confusion)
    print("花费时间：",get_time_dif(start_time))


if __name__ == '__main__':
    # 输入参数定义
    parser = argparse.ArgumentParser(description="bert中文分类")
    parser.add_argument('--use_cuda', default='true', required=True, type=str, help='是否使用GPU')
    parser.add_argument('--train_path', default="", required=True, type=str, help='训练文件')
    parser.add_argument('--dev_path', default='', required=True, type=str, help='验证文件')
    parser.add_argument('--test_path', default='', required=True, type=str, help='测试文件')
    parser.add_argument('--save_path', default='./model_save/', required=True, type=str, help='模型保存路径')
    parser.add_argument('--num_class', default=2, required=True, type=int, help='类别个数')
    parser.add_argument('--num_epoch', default=3, required=True, type=int, help='epoch次数')
    parser.add_argument('--pad_size', default=128, required=True, type=int, help='最大序列长度')
    parser.add_argument('--lr', default=5e-5, required=True, type=float, help='学习率')
    parser.add_argument('--pretrain_model_path', default="", required=True, type=str, help='预训练模型路径')
    parser.add_argument('--batch_size', default=8, required=True, type=int, help='batch大小')

    args = parser.parse_args()
    start_time = time.time()
    config=Config.Config(args)
    print("预训练模型加载".center(40, "_"))
    model = Model(config)
    print("加载模型花费时间".center(40, "_"))
    print("{}秒".format(get_time_dif(start_time)))
    print("训练集数据加载".center(40, "_"))
    train_data_set = TrainData()
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    print("验证集数据加载".center(40, "_"))
    dev_data_set = DevData()
    dev_data_loader = DataLoader(dataset=dev_data_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    print("测试集数据加载".center(40, "_"))
    test_data_set = TestData()
    test_data_loader = DataLoader(dataset=test_data_set, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    train(config, model, train_data_loader, dev_data_loader, test_data_loader)

