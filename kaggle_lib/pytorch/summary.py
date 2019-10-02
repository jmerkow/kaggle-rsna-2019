import os
from datetime import datetime

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid


class TensorboardSummary(object):
    def __init__(self, directory, subdir=None):

        if subdir is None:
            n = datetime.now()
            subdir = 'events_' + n.strftime('%Y%b%d_%H%M%S')

        self.directory = directory
        self.subdir = subdir
        self.logdir = os.path.join(self.directory, self.subdir)
        self.writer = self.create_summary()

    def create_summary(self):
        writer = SummaryWriter(log_dir=self.logdir)
        return writer

    def visualize_image(self, images, preds, masks, epoch, scores=None, ):

        grid_image = make_grid(images, len(images), normalize=True)
        self.writer.add_image('Image', grid_image, epoch)
        grid_image = make_grid(preds, len(images), normalize=False, range=(0, 255))

        text = 'Predicted'
        if scores:
            text += 'Scores: ' + ', '.join("{.4f}".format(v) for v in scores)

        self.writer.add_image(text, grid_image, epoch)
        grid_image = make_grid(masks, len(images), normalize=False, range=(0, 255))
        self.writer.add_image('Mask', grid_image, epoch)

    def add_scalar(self, *args, **kwargs):
        self.writer.add_scalar(*args, **kwargs)
