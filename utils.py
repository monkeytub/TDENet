import torch
import os


class SaveBestModel:
    def __init__(self, model, epoch, epochs, monitor_value, checkpoint_path, best=None) -> None:
        self.epoch = epoch
        self.epochs = epochs
        self.monitor_value = monitor_value
        self.best = best
        self.checkpoin_path = checkpoint_path
        self.model = model

    def run(self):
        # Save best model only
        if self.epoch == 0:
            # print(f'monitor value is: {self.monitor_value:.4f}')
            self.best = self.monitor_value
            # 每一epoch保存一次
            save_dir = os.path.join(self.checkpoin_path, 'best_SNR_0_rand7.pt')
            torch.save(self.model, save_dir)
            print('saved model')
        elif self.best > self.monitor_value:
            # print(f'monitor value is: {self.monitor_value:.4f}')
            self.best = max(self.best, self.monitor_value)
            save_dir = os.path.join(self.checkpoin_path, 'best_SNR_0_rand7.pt')
            torch.save(self.model, save_dir)
            print('saved model')
        elif (self.epoch + 1) == self.epochs:
            save_dir = os.path.join(self.checkpoin_path, 'best_SNR_0_rand7.pt')
            torch.save(self.model, save_dir)
            print('saved model last')
        else:
            pass
