import glob
import logging
import os
import shutil

import torch

logger = logging.getLogger(__name__)


class Saver(object):

    def __init__(self, workdir='.', subdir='checkpoints'):
        self.workdir = os.path.abspath(workdir)
        self.directory = os.path.join(self.workdir, subdir)
        logger.info("saver at '%s'", self.directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=False, **state):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.directory, filename)
        torch.save(state, filename)
        logger.info("saving to %s", filename.replace(self.workdir, '').strip('/'))
        if is_best:
            best_filename = os.path.join(self.directory, 'best_model.pth.tar')
            shutil.copy(filename, best_filename, follow_symlinks=False)
            logger.info("Best Model! Copying to '%s'", best_filename.replace(self.workdir, '').strip('/'))

    def get_checkpoints(self, filename_glob):
        return glob.glob(os.path.join(self.directory, filename_glob))
