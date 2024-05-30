import pytorch_lightning as pl
import Config


class test_step(pl.LightningModule):
    def __init__(self,model):
        super(test_step, self).__init__()
        self.model=model
        self.args = Config.parse_args()