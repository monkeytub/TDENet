import torch
import utils
import argparse
from model import ResConvDeconv
from dataset.my_dataset import CustomDataset
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary


def train(model, train_loader, device, loss_fn, optimizer, train_total, epoch):

    model.train()#运行 model.train() 之后，就告诉了 BN 层，对之后输入的每个 batch 独立计算其均值和方差，BN 层的参数是在不断变化的。

    train_loss = 0.
    train_acc = 0.
    total = 0
    for i, (XYImage, Height) in enumerate(train_loader):

        # Forward
        input_ = XYImage.to(device).float()
        target = Height.to(device).float()
        preds = model(input_)

        # Compute loss
        loss = loss_fn(preds, target)
        train_loss += loss.item()

        # Backward        
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()
        # calculate accuracy, not implemented

        batch_size = target.size(0)
        total += batch_size

    train_acc = train_acc / train_total
    train_loss = train_loss / len(train_loader)
    
    return train_acc, train_loss


def valid(model, val_loader, device, loss_fn, optimizer, valid_total, epoch):
    valid_acc = 0.
    valid_loss = 0.
    model.eval()
    '''可行的方法：将训练过程中每个 batch 的 μ \muμ 和 σ \sigmaσ 都保存下来，然后加权平均当做整个训练数据集的 μ \muμ 和 σ \sigmaσ ，同时用于测试。

model.eval() 就是告诉 BN 层，我现在要测试了，你用刚刚统计的 μ \muμ 和 σ \sigmaσ 来测试我，不要再变了。
————————————————
'''
    with torch.no_grad():
        for i, (XYImage, Height) in enumerate(val_loader):

            input_ = XYImage.to(device).float()
            target = Height.to(device).long()

            preds = model(input_)

            loss = loss_fn(preds, target)
            valid_loss += loss.item()

            # calculate accuracy, not implemented
            
    valid_acc = valid_acc / valid_total
    valid_loss = valid_loss / len(val_loader)

    return valid_acc, valid_loss


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data', help='data_dir')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--epochs', type=int, default=200, help='total epochs to train model, no early stop')
    parser.add_argument('--initial_lr', type=float, default=3e-4, help='initial_lr')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--save_path', type=str, default='./checkpoints/')
    parser.add_argument('--val_percent', type=float, default=0.2)
    args = parser.parse_args()
    print(args)

    epochs = args.epochs
    data_dir = args.data_dir
    initial_lr = args.initial_lr
    batch_size = args.batch_size
    checkpoint_path = args.save_path
    val_percent = args.val_percent

    save_best_model = utils.SaveBestModel(model=None, epoch=None, epochs=args.epochs, monitor_value=None, checkpoint_path=checkpoint_path, best=None)

    dataset = CustomDataset(data_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train_dataset, valid_dataset = random_split(dataset, [n_train, n_val])
    '''train, test = torch.utils.data.random_split(dataset= all_dataset, lengths=[参数1，参数2])
    lengths是一个list，按照对应的数量返回数据个数'''
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=args.num_workers)
    # for i, (XYImage, Height) in enumerate(train_loader):
    #     print(i,(XYImage, Height))


    model = ResConvDeconv.ResidualConvDeconv(1, 1)

    # Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Loss function
    loss_fn = torch.nn.MSELoss().to(device)
    # Optimizer
    optimizer = torch.optim.Adam(lr=initial_lr, params=model.parameters(), betas=(0.9, 0.99))
    # Learning rate decay
    optimizer_step = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

    log_writer = SummaryWriter(comment='Test')

    train_total = len(train_dataset)
    valid_total = len(valid_dataset)

    fake_image = torch.randn((batch_size, 1, 56, 15000)).to(device)
    # print(summary(model, (1, 512, 512), batch_size=1))
    log_writer.add_graph(model, fake_image)

    for epoch in range(epochs):
        train_acc, train_loss = train(model, 
                                      train_loader=train_loader, 
                                      device=device, 
                                      loss_fn=loss_fn,
                                      optimizer=optimizer,
                                      train_total=train_total,
                                      epoch=epoch)
        print(f'Train -> Epoch: {epoch:>03d}, train_acc: {train_acc:.4f}, train_loss: {train_loss:.4f}')
        valid_acc, valid_loss = valid(model, 
                                      val_loader=valid_loader, 
                                      device=device, 
                                      loss_fn=loss_fn, 
                                      optimizer=optimizer,
                                      valid_total=valid_total,
                                      epoch=epoch)
        print(f'Valid -> Epoch: {epoch:>03d}, valid_acc: {valid_acc:.4f}, valid_loss: {valid_loss:.4f}')
        optimizer_step.step()  # update learning rate
        lr = optimizer.param_groups[0]['lr']

        # Write log
        log_writer.add_scalar("Train/Train_Acc", train_acc, epoch)
        log_writer.add_scalar("Valid/Val_Acc", valid_acc, epoch)
        log_writer.add_scalar("Train/Train_Loss", train_loss, epoch)
        log_writer.add_scalar("Valid/Val_Loss", valid_loss, epoch)
        log_writer.add_scalar("LR/lr", lr, epoch)
        # Save best model only
        save_best_model.model = model
        save_best_model.epoch = epoch
        save_best_model.monitor_value = valid_loss
        save_best_model.run()
        

