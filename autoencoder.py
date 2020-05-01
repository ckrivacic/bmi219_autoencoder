"""
Usage:
    autoencoder.py [options]

Options:
    --cuda  Use GPU (not working yet :( )
    --lr=FLOAT, -l  Learning rate  [default: 0.001]
    --layers=STR  comma-separated list of layer sizes  [default: 1000,500,250,2,250,500,1000]
    --fn=STR, -f  which function to use in hidden layers  [default: relu]
    --epochs=INT, -e  Number of epochs  [default: 10]
    --loss=STR, -lf  Which loss function to use  [default: MSELoss]
    --compare=STR, -c  Compare to another loss function
    --outfolder=STR, -o  Where to save everything  [default: default_layers]
    --track, -t  Track progress of image reconstruction
    --batch_size=NUM  Change the batch size  [default: 128]
    --load_df=PATH, -df  load a previously generated df of models
"""
from functions import *
import os

if __name__=='__main__':
    args = docopt.docopt(__doc__)
    layers = str_to_int_list(args['--layers'])
    print('Layers used:')
    print(layers)
    use_cuda = args['--cuda']
    fn = args['--fn']
    print('Using nonlinear function {}'.format(fn))
    lossfn = args['--loss']
    print('Using {} loss function'.format(lossfn))
    outfolder = args['--outfolder']
    df = None
    path = args['--load_df']
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)
    if path:
        import pickle as pkl
        with open(path, 'rb') as f:
            df = pkl.load(f)
        best_index = df['loss'].idxmin()
        best_model = df.at[best_index, 'model']
    #print(df['model'])
    BATCH_SIZE=int(args['--batch_size'])
    NUM_WORKERS=8
    #use_cuda=False
    device = torch.device('cuda' if use_cuda else 'cpu')
    torch.manual_seed(7)

    train_dataset, test_dataset = dataset()

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS)

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS)
    '''
    for tup in iter(train_loader):
        tup[1] = tup[0]
    print(next(iter(train_loader)))
    '''

    if df is None:
        net = Autoencoder()
        if use_cuda:
            net.cuda()
        net.construct_net(layers, fn, device)
        #net.construct_net([1000,2,100], 'sigmoid')

        optimizer = optim.Adam(net.parameters(),
                lr = float(args['--lr']))
                #momentum=True)
        criterion = getattr(nn, lossfn)()

        if args['--compare']:
            compare_criterion = args['--compare']
        else:
            compare_criterion = None

        epochs = int(args['--epochs'])
        best_loss = 100
        rows = []
        for epoch in range(epochs):
            loss = train(net, device,
                    train_loader,
                    optimizer,
                    criterion,
                    epoch, compare=compare_criterion)
            if loss < best_loss: 
                best_loss = loss
                best_model = copy.deepcopy(net)
            row = {}
            row['epoch'] = epoch
            if epoch%10==0: 
                #Before I was saving every epoch but those files
                # were getting huge
                row['model'] = copy.deepcopy(net)
            else:
                row['model'] = None
            row['loss'] = evaluate(net, device, train_loader, criterion)
            row['test'] = evaluate(net, device, test_loader, criterion)
            rows.append(row)

        rows.append({'epoch': epoch + 1, 'model': best_model, 
            'loss': evaluate(best_model, device, train_loader,
                criterion),
            'test': evaluate(best_model, device, test_loader, criterion)})

        df = pd.DataFrame(rows)
        df.to_pickle(os.path.join(outfolder,'epochs.pkl'))

    if args['--track']:
        print(best_model)
        plot_reconstructions(best_model, train_loader, device,
                out=os.path.join(outfolder, 'train_data_tracking',
                    'training_reconstructions_best.png'))

        # Get batch from training data
        batch, classes = next(iter(test_loader))
        # Make a grid from batch
        out = torchvision.utils.make_grid(batch)
        imshow(out, title='Input data', 
                outfile=os.path.join(outfolder, 'test_data_tracking',
                    'input_data.png'))
        for idx, row in df.iterrows():
            epoch = row['epoch']
            if epoch%10==0:
                model = row['model']
                epoch = row['epoch']
                plot_reconstructions(model, train_loader, device,
                        out=os.path.join(outfolder, 'train_data_tracking',
                            'training_reconstructions_epoch_{}.png'.format(epoch)))
                out = reconstructions_from_batch(model, batch, device)
                out = torchvision.utils.make_grid(out)
                imshow(out, title='Test data, epoch {}'.format(epoch),
                        outfile=os.path.join(outfolder,
                            'test_data_tracking',
                            'imshow_epoch_{}.png'.format(epoch)))

    loss_curve(df)
    predictions, labels = encode(best_model, device, test_loader)
    plot_latentspace(predictions, labels, best_model, device)
