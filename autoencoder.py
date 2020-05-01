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
"""
from functions import *
import os

if __name__=='__main__':
    args = docopt.docopt(__doc__)
    layers = str_to_int_list(args['--layers'])
    use_cuda = args['--cuda']
    fn = args['--fn']
    lossfn = args['--loss']
    outfolder = args['--outfolder']
    if not os.path.exists(outfolder):
        os.makedirs(outfolder, exist_ok=True)
    BATCH_SIZE=128
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
        row['model'] = copy.deepcopy(net)
        row['loss'] = evaluate(net, device, train_loader, criterion)
        row['test'] = evaluate(net, device, test_loader, criterion)
        rows.append(row)
        if args['--track']:
            plot_reconstructions(net, train_loader, device,
                    out=os.path.join(outfolder,
                        'training_reconstructions_epoch_{}.png'.format(epoch)))

    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(outfolder,'epochs.pkl'))

    if args['--track']:
        plot_reconstructions(best_model, train_loader, device,
                out=os.path.join(outfolder,
                    'training_reconstructions_best.png'))

        # Get batch from training data
        batch, classes = next(iter(test_loader))
        # Make a grid from batch
        out = torchvision.utils.make_grid(batch)
        imshow(out, title='Input data')
        for idx, row in df.iterrows():
            model = row['model']
            epoch = row['epoch']
            out = reconstructions_from_batch(model, batch, device)
            out = torchvision.utils.make_grid(out)
            imshow(out, title='Test data, epoch {}'.format(epoch))

    loss_curve(df)
    predictions, labels = encode(net, device, test_loader)
    plot_latentspace(predictions, labels, net, device)
