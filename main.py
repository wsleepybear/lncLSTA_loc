
#优化代码
from model.bilstm_gcn import lncG
from train_step import model_step
import dataset
import Config
import os
from utils import *
from torch.utils.data import DataLoader,Subset
import matplotlib
from MetricTracker import MetricTracker
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
matplotlib.use('Agg')
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold


#   train_dataset=dataset.RNAdataset("prepared/lncRNA copy.csv")
def main():

    args = Config.parse_args() #参数

    data=dataset.RNAdataset(args.RNA_data)#lncRNA copy.csv
    num_label=data.num_label#标签
    k=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    index=0
    nowtime=time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    path=os.path.join(args.save,nowtime)
    os.mkdir(path, mode=0o775 )
    # init logger
    logger = init_logger()
    model=lncG()
    logger.info(model)
    record=pd.DataFrame(data=None,columns=['miAUC','maAUC','acc','f1'])
    record.to_csv(path+'/record.csv',index=None)
    record.to_csv('result/record.csv',index=None)
    
    for train_index, test_index in k.split(data,num_label):#5折交叉验证
        model_path = os.path.join(path, 'model.pth%s'%index)
        train_dataset, test_dataset = Subset(data,train_index),Subset(data,test_index)
        train_loader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate_fn)
        test_loader=DataLoader(test_dataset,batch_size=args.test_batch_size,collate_fn=collate_fn)
        metricTracker = MetricTracker()#指标
        model=lncG()#  model=lncG()
        classifier=model_step(path,model,logger)#模型
        trainer = pl.Trainer(gpus =1,  
                            max_epochs=args.epochs,
                            callbacks=[metricTracker,
                            EarlyStopping(monitor="loss", mode="max", patience=60),
                            ModelCheckpoint(save_top_k=1, monitor="f1", mode="max", save_on_train_epoch_end=False)])#训练
        
        trainer.fit(classifier,train_loader,test_loader)#训练
        trainer.test(dataloaders=test_loader)#测试
        model = classifier.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,path=path,model=model,logger=logger)#加载最好的模型
        torch.save(model.model.state_dict(),model_path)#保存模型
        index+=1
    
    result=pd.read_csv(path+'/record.csv')
    miAUC_mean=np.average(result['miAUC'])
    maAUC_mean=np.average(result['maAUC'])
    acc_mean=np.average(result['acc'])
    f1_mean=np.average(result['f1'])
    result.loc[result.shape[0]]=[miAUC_mean,maAUC_mean,acc_mean,f1_mean]
    logger.info(result)
    result.to_csv(path+'/record.csv',index=None)
    result.to_csv('result/record%s.csv'%i,index=None)

if __name__ == "__main__":
    main()

