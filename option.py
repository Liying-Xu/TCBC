import torch
import argparse


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--input_size', type=int, default=256, help='input size')
        parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser
        return parser.parse_args()


    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.opt = opt
        return self.opt

class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument("--local_rank", default=-1)
        parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
        parser.add_argument('--decay_epoch', type=int, default=100,help='epoch to start linearly decaying the learning rate to 0')
        parser.add_argument('--train_epoch', type=int, default=200, help='number of train epochs')
        parser.add_argument('--phase', type=str, default='UNPAIRED', help='train, val, test, etc')
        parser.add_argument('--dataset', required=False, default='suzhou', help='')
        parser.add_argument('--save_root', required=False, default='results', help='results save path')
        parser.add_argument('--dataroot', required=False, default='G:\XLY\DATA\XLY_DATA', help='数据集加载路径')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--batch_size', type=int, default=3, help='train/test batch size')
        parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
        #parser.add_argument('--ngf', type=int, default=8, help='Gg的基础特征数')
        #parser.add_argument('--naf', type=int, default=8, help='Gc的基础特征数')
        #parser.add_argument('--ndf', type=int, default=8, help='D的基础特征数')
        parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
        parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
        parser.add_argument('--beta1', type=float, default=0.5, help='c for Adam optimizer')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
        self.isTrain = True
        return parser

class TestOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--dataset', required=False, default='suzhou', help='')
        parser.add_argument('--dataroot', required=False, default='G:\XLY\DATA\XLY_DATA', help='数据集加载路径')
        parser.add_argument('--phase', required=False, default='UNPAIRED', help='')
        parser.add_argument('--ngf', type=int, default=64)
        parser.add_argument('--save_root', required=False, default='results', help='results save path')
        parser.add_argument('--n_epoch',type=int, default=200, help='load epoch number')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.isTrain = True
        return parser