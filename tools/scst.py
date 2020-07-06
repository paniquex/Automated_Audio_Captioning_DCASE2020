import torch
from torch import nn
#from processes.method import decode_outputs
from eval_metrics import evaluate_spider

from typing import MutableMapping, MutableSequence,\
    Any, Union, List, Dict, Tuple
from torch import Tensor, no_grad, save as pt_save, \
    load as pt_load, randperm
from loguru import logger
from pathlib import Path
from torch.nn.functional import softmax

__author__ = "Nikita Kuzmin -- Lomonosov Moscow State University"


def _decode_outputs(predicted_outputs: MutableSequence[Tensor],
                    ground_truth_outputs: MutableSequence[Tensor],
                    indices_object: MutableSequence[str],
                    file_names: MutableSequence[Path],
                    eos_token: str,
                    print_to_console: bool) \
        -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Decodes predicted output to string.

    :param predicted_outputs: Predicted outputs.
    :type predicted_outputs: list[torch.Tensor]
    :param ground_truth_outputs: Ground truth outputs.
    :type ground_truth_outputs: list[torch.Tensor]
    :param indices_object: Object to map indices to text (words or chars).
    :type indices_object: list[str]
    :param file_names: List of ile names used.
    :type file_names: list[pathlib.Path]
    :param eos_token: End of sequence token to be used.
    :type eos_token: str
    :param print_to_console: Print captions to console?
    :type print_to_console: bool
    :return: Predicted and ground truth captions for scoring.
    :rtype: (list[dict[str, str]], list[dict[str, str]])
    """
    caption_logger = logger.bind(is_caption=True, indent=None)
    main_logger = logger.bind(is_caption=False, indent=0)
    caption_logger.info('Captions start')
    main_logger.info('Starting decoding of captions')
    text_sep = '-' * 100

    captions_pred: List[Dict] = []
    captions_gt: List[Dict] = []
    f_names: List[str] = []

    if print_to_console:
        main_logger.info(f'{text_sep}\n{text_sep}\n{text_sep}\n\n')

    for gt_words, b_predictions, f_name in zip(
            ground_truth_outputs, predicted_outputs, file_names):
        predicted_words = softmax(b_predictions, dim=-1).argmax(1)

        predicted_caption = [indices_object[i.item()]
                             for i in predicted_words]
        gt_caption = [indices_object[i.item()]
                      for i in gt_words]

        gt_caption = gt_caption[:gt_caption.index(eos_token)]
        try:
            predicted_caption = predicted_caption[
                                :predicted_caption.index(eos_token)]
        except ValueError:
            pass

        predicted_caption = ' '.join(predicted_caption)
        gt_caption = ' '.join(gt_caption)

        f_n = f_name.stem.split('.')[0]

        if f_n not in f_names:
            f_names.append(f_n)
            captions_pred.append({
                'file_name': f_n,
                'caption_predicted': predicted_caption})
            captions_gt.append({
                'file_name': f_n,
                'caption_1': gt_caption})
        else:
            for d_i, d in enumerate(captions_gt):
                if f_n == d['file_name']:
                    len_captions = len([i_c for i_c in d.keys()
                                        if i_c.startswith('caption_')]) + 1
                    d.update({f'caption_{len_captions}': gt_caption})
                    captions_gt[d_i] = d
                    break

        log_strings = [f'Captions for file {f_name.stem}: ',
                       f'\tPredicted caption: {predicted_caption}',
                       f'\tOriginal caption: {gt_caption}\n\n']
        if print_to_console:
            [caption_logger.info(log_string)
             for log_string in log_strings]

        if print_to_console:
            [main_logger.info(log_string)
             for log_string in log_strings]

    if print_to_console:
        main_logger.info(f'{text_sep}\n{text_sep}\n{text_sep}\n\n')

    logger.bind(is_caption=False, indent=0).info(
        'Decoding of captions ended')

    return captions_pred, captions_gt


class ScstTrainer:
    def __init__(self, metric, reg_coef=0.01, terminal_token_id=9):
        self.metric = metric
        self.reg_coef = reg_coef
        self.terminal_token_id = terminal_token_id

    def __call__(self, log_probs, y, fnames):
        # shape: (batch, seq_len, dict_size)
        probs = torch.exp(log_probs)
        print('a')
        baseline_actions = log_probs#torch.argmax(log_probs, dim=-1)
        print('b')
        baseline_score = self.metric([baseline_actions.detach().cpu()], [y.detach().cpu()], fnames)
        print('c')
        probs_reshaped = probs.view(-1, probs.shape[2])
        actions = torch.multinomial(
            probs_reshaped, num_samples=1).squeeze(1)
        # shape: (b, seq_len)
        actions = actions.reshape(*probs.shape[:2])
        score = self.metric(actions, y, fnames)
        # shape: (b, seq_len)
        log_probs_for_actions = torch.gather(
            log_probs, -1, actions.unsqueeze(-1)
        ).squeeze(-1)

        # handling terminal tokens
        # https://discuss.pytorch.org/t/first-nonzero-index/24769/2
        terminal_mask = actions == self.terminal_token_id
        vals, idx = terminal_mask.max(dim=-1)
        # first terminal token is agent's action
        idx = idx + 1
        # if terminal token did not occur
        idx[vals == 0] = actions.shape[1]

        # https://discuss.pytorch.org/t/filling-torch-tensor-with-zeros-after-certain-index/64842
        invalid_mask = torch.arange(actions.shape[1]) >= idx.unsqueeze(-1)
        log_probs_for_actions[invalid_mask] = 1.

        # shape: (batch,)
        log_probs_for_trajectories = log_probs_for_actions.sum(dim=1)
        assert not baseline_score.requires_grad
        assert not score.requires_grad
        assert baseline_score.dim() == score.dim() \
            == log_probs_for_trajectories.dim() == 1
        assert baseline_score.shape[0] == score.shape[0] \
            == log_probs_for_trajectories.shape[0]

        advantage = score - baseline_score
        J = advantage * log_probs_for_trajectories
        # shape: (batch, seq_len)
        entropy = -(log_probs * probs).sum(-1)
        # shape: (batch,)
        entropy = entropy.mean(-1)

        # both maximized
        loss = -J - entropy
        return loss


class SpiderLoss():
    def __init__(self, indices_list):
        self.indices_list = indices_list

    def __call__(self, y_hat, y, fnames):
        captions_pred, captions_gt = _decode_outputs(
            y_hat,
            y,
            indices_object=self.indices_list,
            file_names=sorted(fnames),
            eos_token='<eos>',
            print_to_console=False)

        metrics = evaluate_spider(captions_pred, captions_gt,
                                   verbose=False)
        return metrics.items()[-1][1]


def test():
    torch.manual_seed(17)
    device = torch.device('cpu')

    def metric(x, y):
        diff = x - y
        diff = diff.float()
        diff = torch.mean(diff ** 2, dim=-1)
        return diff


    trainer = ScstTrainer(metric, reg_coef=0.01, terminal_token_id=0)

    b = 2
    n_actions = 3
    t = 5

    log_probs = torch.rand(b, t, n_actions, device=device)
    y = torch.randint(n_actions, (b, t), device=device)

    loss = trainer.eval_loss(log_probs, y)
    print(loss)


if __name__ == "__main__":
    test()