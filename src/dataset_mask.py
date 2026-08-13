"""Build FakeSV+ from FakeSV (Sec. 4.1 of the paper).

FakeSV+ extends FakeSV [Qi et al., AAAI 2023] into a **modality-incomplete**
benchmark.  Given a global missing rate :math:`\\eta`, every sample may lose an
arbitrary subset of its modalities, so that all :math:`2^M - 1` combinations
(:math:`M = 3`) can occur while at least one modality is always kept::

    eta = 1 - (sum_i m_i) / (L * M)                                   (Eq. 15)

where :math:`m_i` is the number of available modalities of the :math:`i`-th
sample, :math:`L` the number of samples and :math:`M` the number of modalities.

The script writes

* ``<out>/embs/<vid>.pkl`` -- features with the missing modalities zero-padded
  (Eq. 1);
* ``<out>/masker.json`` -- ``{vid: {vision: 0/1, text: 0/1, audio: 0/1}}`` where
  ``1`` means *missing*.  ``utils/dataset.py`` turns these flags into the 4-digit
  missing prompt of Eq. 2;
* ``<out>/statistics.json`` -- the realised combination distribution, i.e. the
  numbers reported in Table 2.

Usage::

    python dataset_mask.py --dataset FakeSV --missing_rate 0.3 --seed 2024
"""

import argparse
import json
import os
import pickle
import random
import shutil

import torch

MODALITIES = ('vision', 'text', 'audio')
COMBINATION_NAMES = {
    (1, 0, 0): 'Video',
    (0, 1, 0): 'Text',
    (0, 0, 1): 'Audio',
    (0, 1, 1): 'Text+Audio',
    (1, 0, 1): 'Video+Audio',
    (1, 1, 0): 'Text+Video',
    (1, 1, 1): 'Text+Audio+Video',
}


class FakeSVPlusMasker:
    """Sample a missing pattern per video and materialise the masked features."""

    def __init__(self, config):
        self.source_emb_dir = config['source_emb_dir']
        self.target_dir = config['target_dir']
        self.target_emb_dir = os.path.join(self.target_dir, 'embs')
        self.missing_rate = config['missing_rate']
        self.masker_fp = config['masker_fp']
        self.copy_annotation = config['copy_annotation']
        self.source_dir = config['source_dir']

    # ---------------------------------------------------------------- sampling
    def _sample_pattern(self):
        """Draw a missing pattern with at least one available modality."""
        while True:
            pattern = {m: int(random.random() <= self.missing_rate) for m in MODALITIES}
            if sum(pattern.values()) < len(MODALITIES):
                return pattern

    def _rebalance(self, masker):
        """Compensate the bias introduced by the "at least one" constraint.

        Rejection sampling lowers the realised missing rate.  We therefore keep
        flipping randomly chosen available modalities until the realised rate of
        Eq. 15 reaches the target (an exact match is impossible because the rate
        is quantised by ``1 / (L * M)``).
        """
        total_slots = len(masker) * len(MODALITIES)
        target_missing = int(round(self.missing_rate * total_slots))
        current_missing = sum(sum(p.values()) for p in masker.values())

        vids = list(masker.keys())
        guard = 0
        while current_missing < target_missing and guard < total_slots * 10:
            guard += 1
            vid = random.choice(vids)
            pattern = masker[vid]
            if sum(pattern.values()) >= len(MODALITIES) - 1:
                continue  # keep at least one available modality
            available = [m for m in MODALITIES if pattern[m] == 0]
            pattern[random.choice(available)] = 1
            current_missing += 1
        return masker

    def build_masker(self, vids):
        masker = {vid: self._sample_pattern() for vid in vids}
        if self.missing_rate > 0:
            masker = self._rebalance(masker)
        return masker

    # ------------------------------------------------------------------ output
    def statistics(self, masker):
        total_slots = len(masker) * len(MODALITIES)
        missing = sum(sum(p.values()) for p in masker.values())
        distribution = {name: 0 for name in COMBINATION_NAMES.values()}
        for pattern in masker.values():
            key = tuple(1 - pattern[m] for m in ('vision', 'text', 'audio'))
            distribution[COMBINATION_NAMES[key]] += 1
        num = max(len(masker), 1)
        return {
            'num_videos': len(masker),
            'target_missing_rate': self.missing_rate,
            'actual_missing_rate': missing / max(total_slots, 1),
            'combination_distribution(%)': {
                k: round(v / num * 100, 2) for k, v in distribution.items()
            },
        }

    def masking(self):
        os.makedirs(self.target_emb_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(self.source_emb_dir) if f.endswith('.pkl'))
        if not files:
            raise FileNotFoundError('No .pkl feature found in ' + self.source_emb_dir)
        vids = [f.rsplit('.', 1)[0] for f in files]

        masker = self.build_masker(vids)

        for filename, vid in zip(files, vids):
            with open(os.path.join(self.source_emb_dir, filename), 'rb') as f:
                data = pickle.load(f)
            pattern = masker[vid]
            for modality in MODALITIES:
                if pattern[modality] == 1 and modality in data:
                    # Eq. 1: zero padding for the modalities tagged as missing.
                    data[modality] = torch.zeros_like(torch.as_tensor(data[modality]))
            with open(os.path.join(self.target_emb_dir, filename), 'wb') as f:
                pickle.dump(data, f)

        with open(self.masker_fp, 'w') as f:
            json.dump({vid + '.pkl': masker[vid] for vid in vids}, f, indent=2)

        stats = self.statistics(masker)
        with open(os.path.join(self.target_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        if self.copy_annotation:
            src = os.path.join(self.source_dir, 'data')
            dst = os.path.join(self.target_dir, 'data')
            if os.path.isdir(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)

        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print('Masker saved to ' + self.masker_fp)


def build_config(args):
    source_dir = os.path.join(args.data_root, args.dataset)
    source_emb_dir = os.path.join(source_dir, 'embs')
    if not os.path.isdir(source_emb_dir):
        raise FileNotFoundError('Expected FakeSV features at ' + source_emb_dir)

    suffix = str(int(round(args.missing_rate * 100)))
    target_dir = os.path.join(args.data_root, '{}-{}'.format(args.dataset, suffix))
    if os.path.exists(target_dir) and not args.overwrite:
        raise FileExistsError(
            '{} already exists, pass --overwrite to regenerate it.'.format(target_dir)
        )
    os.makedirs(target_dir, exist_ok=True)

    return {
        'source_dir': source_dir,
        'source_emb_dir': source_emb_dir,
        'target_dir': target_dir,
        'missing_rate': args.missing_rate,
        'masker_fp': os.path.join(target_dir, args.masker_fp),
        'copy_annotation': args.copy_annotation,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the FakeSV+ dataset.')
    parser.add_argument('--dataset', type=str, default='FakeSV', help='source dataset name')
    parser.add_argument('--data_root', type=str, default='../dataset/', help='dataset root dir')
    parser.add_argument('--missing_rate', '--mask_ratio', dest='missing_rate', type=float,
                        default=0.3, help='global missing rate eta of Eq. 15')
    parser.add_argument('--masker_fp', type=str, default='masker.json')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--overwrite', action='store_true', help='overwrite an existing output dir')
    parser.add_argument('--copy_annotation', action='store_true', default=True,
                        help='copy data/ (annotations and split files) to the new dataset dir')
    args, _ = parser.parse_known_args()

    if not 0.0 <= args.missing_rate < 1.0:
        raise ValueError('missing_rate must lie in [0, 1)')

    random.seed(args.seed)
    FakeSVPlusMasker(build_config(args)).masking()
