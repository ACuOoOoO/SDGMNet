#import Losses
import PhotoTour
from models import SDGMNet
import torch
import EvalMetrics
import os
from Utils import transform_test
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def CreateTestLoader(name='yosemite'):
    TestData = PhotoTour.PhotoTour(root = '/data1/ACuO/UBC', name = name, download = False, train = False, transform = transform_test)
    Testloader = torch.utils.data.DataLoader(TestData, batch_size = 1024,
                                                shuffle=False, num_workers=32, pin_memory = True)
    return Testloader

def test(model, testloader, GPU=True):
    model.eval()
    with torch.no_grad():
        if GPU:
            simi = torch.zeros(0).cuda()
        else:
            simi = torch.zeros(0)
        lbl = torch.zeros(0)
        for i, (data1, data2, m) in enumerate(testloader):
            if GPU:
                data1 = data1.cuda(non_blocking=True)
                data2 = data2.cuda(non_blocking=True)
            t1 = model(data1)
            t2 = model(data2)
            t3 = torch.sum(t1*t2, dim=1).detach().view(-1)
            simi = torch.cat((simi, t3), dim=0)
            lbl = torch.cat((lbl, m.view(-1)), dim=0)
        lbl = lbl.numpy()
        simi = simi.cpu().numpy()
        FPR = EvalMetrics.ErrorRateAt95Recall(labels=lbl, scores=simi+10)
    return FPR


pretrained_model = './pretrained/models/model_no.pt'
model = SDGMNet()
checkpoint = torch.load(pretrained_model)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
TestLoader1 = CreateTestLoader('liberty')
TestLoader2 = CreateTestLoader('yosemite')

FPR1 = test(model,TestLoader1)
FPR2 = test(model,TestLoader2)

print((FPR1+FPR2)/2)
