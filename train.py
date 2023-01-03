from util.utils import ConfigDict, PRNGSequence
from util.scms import gen_scm
from genDM.configs import train_config, handle_config
from util.dataset import JaxDataset, NumpyLoader

from genDM.trainer import MLETrainer, MinMaxTrainer
from genDM.losses import y2_loss
from genDM.models import CPM

import tqdm
import matplotlib.pyplot as plt


def train(config: ConfigDict):
    # Load data
    rng = PRNGSequence(config.gc.seed)
    scm = gen_scm(config.gc.dataset)
    config = handle_config(config, scm)  # Update config based on the data
    dl = NumpyLoader(JaxDataset(scm.generate(next(rng), config.gc.n_samples)), batch_size=config.gc.batch_size)

    # MLE for log p_true(t|x)
    mle_trainer = MLETrainer(config.gc, config.mle_config, config.doc)
    mle_trainer.initialize(next(rng))
    with tqdm.tqdm(total=config.gc.MLE_epochs) as pbar:
        for epoch in range(config.gc.MLE_epochs):
            for batch in dl:
                outputs = mle_trainer.train_step(next(rng), batch)
            logging_str = ' '.join(
                ['{}: {: <10g}'.format(k, v) for (k, v) in outputs.items()]) if outputs is not None else ''
            pbar.set_postfix_str(logging_str)
            pbar.update(1)

    # MinMax for log pi_theta(t|x)
    target_loss = y2_loss(scm)
    minmax_trainer = MinMaxTrainer(target_loss=target_loss,
                                   log_p_true=mle_trainer.log_p,
                                   global_config=config.gc,
                                   decision_model_config=config.dmc, adversary_model_config=config.amc,
                                   decision_optim_config=config.doc, adversary_optim_config=config.aoc)
    dl = NumpyLoader(JaxDataset(scm.data), batch_size=config.gc.batch_size)  # Reload the data
    minmax_trainer.initialize(next(rng))
    with tqdm.tqdm(total=config.gc.epochs) as pbar:
        with tqdm.tqdm(total=config.gc.epochs) as dbar:
            for epoch in range(config.gc.epochs):
                for batch in dl:
                    outputs, debug = minmax_trainer.train_step(next(rng), batch)
                logging_str = ' '.join(['{}: {: .4f}'.format(k, v) for (k, v) in outputs.items()])
                pbar.set_postfix_str(logging_str)
                pbar.update(1)
                if config.gc.debug:
                    debug_str = ' '.join(['{}: {: .4f}'.format(k, v) for (k, v) in debug.items()])
                    dbar.set_postfix_str(debug_str)
                    dbar.update(1)
                    # TODO 1. The raw mean in ratio model goes to zero,
                    #      2. Check if the learned policy is optimal,
                    #      3. Save and load models

    mu_x = minmax_trainer.decision_state.apply_fn({'params': minmax_trainer.decision_state.params},
                                                  scm.data[:, config.gc.batch_condition_dims],
                                                  method=CPM.mu)
    plt.plot(scm.data[:, config.gc.batch_condition_dims], mu_x, 'o')
    plt.savefig('outputs/learned_policy.png')
    plt.close()


if __name__ == '__main__':
    train(train_config)

