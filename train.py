import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from datetime import datetime

import models
from models import PixelCnn
from dataset import Data
from util import init_argparser, show_samples, save_training_plot

def evaluate(test, model):
    loss = 0
    model.eval()

    for batch in test:
        bsize = batch['y'].shape[0]
        loss += model.get_loss(model(batch['x']), batch['y']).item() * bsize
    
    return loss

def train_batch(batch, model, optimizer):
    optimizer.zero_grad()
    preds = model(batch['x'])
    loss = model.get_loss(preds, batch['y'])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()

    return loss.item()

def training(train, test, model, optimizer, epochs):
    nlls_train = []
    nlls_test = []

    for epoch in range(1, epochs+1):
        model.train()
        ttl_nll = 0
        for batch in train:
            bsize = batch['y'].shape[0]
            nll = train_batch(batch, model, optimizer)
            ttl_nll += nll * bsize
            nlls_train.append(nll)
        
        nlls_test.append(evaluate(test, model)/len(test.dataset))
        print('epoch '+str(epoch), 'train: '+str(ttl_nll/len(train.dataset)), 'test: '+str(nlls_test[epoch-1]))
    
    return np.array(nlls_train), np.array(nlls_test)

def main(args: argparse.Namespace):
    start = datetime.now()
    train_arr, test_arr = Data.read_pickle(args.pickle, 'train'), Data.read_pickle(args.pickle, 'test')
    train = DataLoader(Data(train_arr, args.dev), batch_size=args.bsize, num_workers=args.workers, shuffle=True, collate_fn=Data.collate_fn)
    test = DataLoader(Data(test_arr, args.dev, train.dataset.mean, train.dataset.std), batch_size=args.bsize, num_workers=args.workers,  shuffle=True, collate_fn=Data.collate_fn)
    model = PixelCnn(train.dataset.W, train.dataset.C, args.kernel_size, args.layers, args.filters, args.dist_size, getattr(models, args.conv_class))
    if args.dev=='cuda':
        model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    nlls_train, nlls_test = training(train, test, model, optimizer, args.epochs)
    print((datetime.now()-start).total_seconds())
    samples = model.generate_samples(args.n_samples, args.dev, train.dataset.mean, train.dataset.std).cpu().numpy()

    save_training_plot(nlls_train, nlls_test, 'NLL (nats/dim)', args.nll_img_path)
    show_samples(
        samples.reshape(args.n_samples, train.dataset.H, train.dataset.W, train.dataset.C), 
        args.samples_img_path)

if __name__=='__main__':
    main(init_argparser())

    

