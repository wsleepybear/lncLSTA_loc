import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--RNA_data',
                        default="prepared/lncRNA.csv")
    parser.add_argument('--phy_path',
                        default='./prepared/physicochemical_6.csv')
    parser.add_argument('--isphy',
                        default=False,
                        type=bool)
    parser.add_argument('--method',
                        default='one_hot',
                        type=str)
    parser.add_argument('--phy_num',
                        default=11,
                        type=int)
    parser.add_argument('--k',
                        default=3,
                        type=int)
    parser.add_argument('--save',
                        default='/data/wk/checkpoint/')
    parser.add_argument('--stride',
                        default=1,
                        type=int)
    
    
    # model arguments
    parser.add_argument('--emb_dim',default=64,type=int)
    parser.add_argument('--cnn1_num',default=32,type=int)
    parser.add_argument('--cnn1_kernel',default=5,type=int)
    parser.add_argument('--cnn2_kernel',default=3,type=int)
    parser.add_argument('--spp_size',default=128,type=int)
    parser.add_argument('--spp_size2',default=128,type=int)
    parser.add_argument('--gin',default=36,type=int)
    parser.add_argument('--gout',default=72,type=int)
    parser.add_argument('--lsatt1_head',default=6,type=int)
    parser.add_argument('--num_layer',default=1,type=int)
    parser.add_argument('--lsatt1_window_size',default=128,type=int)
    parser.add_argument('--lsatt1_r',default=4,type=int)
    parser.add_argument('--attdrop',default=0.65,type=int)
    parser.add_argument('--contextSizeList',default=[1,3,5])
    parser.add_argument('--fc1in',default=32,type=int)
    parser.add_argument('--fc3in',default=32,type=int)
    parser.add_argument('--convdrop',default=0.5,type=float)   
    parser.add_argument('--dropout',default=0.5,type=float)  
    

    # training arguments
    parser.add_argument('--epochs',
                        default=1000,
                        type=int)
    parser.add_argument('--batch_size',
                        default=256,
                        type=int)
    parser.add_argument('--test_batch_size',
                        default=512,
                        type=int)
    parser.add_argument('--learning_rate',
                        default=2e-4,
                        type=float)
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float)
                   
    # testing arguments
    parser.add_argument('--model_path',
                        default='checkpoints/model.pth')
    args = parser.parse_known_args()[0]
    return args
