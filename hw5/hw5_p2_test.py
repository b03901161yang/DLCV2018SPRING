from inception_model_noBN import *
from vd_dataloader import  *
import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import sys
import os

def update_progress(progress, loss1):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1:.2f}%: acc = {2:.2f}%".format( "#"*block + "-"*(barLength-block), round(progress*100, 3), loss1*100)
    sys.stdout.write(text)
    sys.stdout.flush()


class actRecog_RNN_VarLen_classifier(nn.Module):
    def __init__(self):
        # input: batch_size, num_frame, 2048
        self.drop_rate = 0.3
        super(actRecog_RNN_VarLen_classifier, self).__init__()
        self.gru = nn.GRU(2048, 256, 1, batch_first=True, dropout=self.drop_rate, bidirectional=True)
        self.fc = nn.Linear(512, 11)
        
    def forward(self, x, hidden=None):
        out, _ = self.gru(x, hidden)
        #print('LSTM output shape:', out.shape)
        out = out[:, -1, :]
        out = self.fc(out)
        return F.log_softmax(out, dim=1)
    
    def forward_gru(self, x, hidden=None):
        _, hidden_out = self.gru(x, hidden)
        return hidden_out
    
    
def testCRNN_VarLen(cnn_model, rnn_model, dataloader, batch_size, guid, filename):

    epoch_loss = 0.
    epoch_acc  = 0.
    num_examples = 0
    file = open(filename, 'w')
   
    assert batch_size == 1
    frame_batch = 8 # simulataneous prediction of frame_batch frames

    for batch_i, sample in enumerate(dataloader):
        x = Variable(sample['X'])
        label = Variable(sample['Y'])
        if guid >= 0:   # move to GPU
            x = x.cuda(guid)
            label = label.cuda(guid)

        # (1, num_frame, 3, H, W) -> (num_frame, 3, H, W)
        x.squeeze_(0)
        #print('x shape:', x.shape)

        # split to different sub-batch
        split_x = torch.split(x, frame_batch, dim=0) # tuple
        feature_list = list()
        for i in range(len(split_x)):
            CNN_feature = cnn_model(split_x[i])
            feature_list.append(CNN_feature)
            
        # (num_frame, 2048)
        temp_feature = torch.cat(feature_list, dim=0) 
        
        # (1, num_frame, 2048)
        temp_feature = temp_feature.unsqueeze_(0)

        #print('Temp feature shape:', temp_feature.shape)
        
        y = rnn_model.forward(temp_feature)
        #print('Predcited label shape:', y.shape)
        #print('Label shape:', label.shape)
        label = label.squeeze_()
    
        
        feature_list = list()
        temp_feature = 0
        split_x = 0
        
        _, preds = torch.max(y.data, 1)

        #print('predict size', preds[0])
        file.write(str(int(preds[0])))
        file.write('\n')
        # statistics
        epoch_acc  = epoch_acc  + torch.sum(preds == label.data)
        num_examples = num_examples + label.shape[0]
        
        update_progress(batch_i/len(dataloader), (epoch_acc/num_examples))
        
        #label = 0
        #y = 0
       
    file.close()
    print(" ")
    print("accuracy: {}\n".format(epoch_acc/num_examples))

    

if __name__ == '__main__':
    save_dir = ''
    batch_size = 1
    guid = 0
    
    data_path = sys.argv[1]
    csv_path = sys.argv[2]
    out_dir = sys.argv[3]
    out_path = os.path.join(out_dir, 'p2_result.txt')
    cnn_model = torch.load('hw5_p2_cnn.pt')
    rnn_model = torch.load('hw5_p2_rnn.pt')
    print(cnn_model)
    print('==============================')
    print(rnn_model)
    cnn_model = cnn_model.cuda(guid)
    rnn_model = rnn_model.cuda(guid)

    # create dataset
    dataset = VdDataset(dir=data_path, csv_file=csv_path)
    
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    testCRNN_VarLen(cnn_model, rnn_model, dataloader, batch_size, guid, out_path)

    
