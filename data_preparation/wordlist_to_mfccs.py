"""Generate MFCC features for words.
"""
from __future__ import division
import argparse
import h5py
from matplotlib import pyplot
import numpy
import os
from scipy.io import wavfile
import subprocess

import mfcc



def main(args):
    words = []
    wordstoindex = {}
    mfcc_convert = mfcc.MFCC()
    mfcc_savepath_prefix = os.path.splitext(args.mfccs)[0]
    num_frames = 0
    h5file = h5py.File(args.mfccs, "w")
    for line_id, line in enumerate(open(args.wordlist, "r").readlines()):
        word, path, start, end = line.strip().split()
        if not wordstoindex.has_key(word):
            wordstoindex[word] = len(words)
            words.append(word)
        class_id = wordstoindex[word]
        start, end = int(start), int(end)
        command = ("sox -t sph {0} -r 16000 -t wav {1}"
                   ).format(path, "tmp.wav")
        subprocess.call(command, shell=True)
        sample_rate, samples = wavfile.read("tmp.wav")
        samples = samples.astype(float)
        word_samples = samples[start - 480: end + 480]
        word_mfccs = mfcc_convert.sig2s2mfc(word_samples)
        n_frames, n_dims = word_mfccs.shape
        full_mfccs = numpy.empty((n_frames, n_dims * 3), dtype=numpy.float32)
        full_mfccs[:,:n_dims] = word_mfccs
        mfcc.deltas(word_mfccs, output_frames=full_mfccs[:,n_dims:2*n_dims])
        mfcc.deltas(full_mfccs[:,n_dims:2*n_dims],
                    output_frames=full_mfccs[:,2*n_dims:])
        if line_id == 0:
            mfcc_dset = h5file.create_dataset("mfccs", (n_frames, 39), maxshape=(None, 39), dtype=numpy.float32)
            label_dset = h5file.create_dataset("labels", (100, 3), maxshape=(None, 3), dtype=numpy.int32)
            data_idx = 0
            cur_idx = 0
        if cur_idx + n_frames >= len(mfcc_dset):
            h5file.flush()
            mfcc_dset.resize((2*len(mfcc_dset), mfcc_dset.shape[1]))
            print "mfcc doubling", cur_idx
        if data_idx == len(label_dset):
            h5file.flush()
            label_dset.resize((2*label_dset.shape[0], label_dset.shape[1]))
        try:
            mfcc_dset[cur_idx:cur_idx+n_frames] = full_mfccs
        except:
            import pdb; pdb.set_trace()
        label_dset[data_idx] = class_id, cur_idx, n_frames
        cur_idx += n_frames
        data_idx += 1
    mfcc_dset.resize((cur_idx, mfcc_dset.shape[1]))
    label_dset.resize((data_idx, label_dset.shape[1]))
    h5file.flush()
    words.sort()
    open(args.wordkey, "w").write("\n".join(
        "%s %d" % (w, wordstoindex[w]) for w in words))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute MFCC features of words")
    parser.add_argument("wordlist", type=str, help="string for where word list "
                        "are saved.")
    parser.add_argument("mfccs", type=str, help="path to save mfcc features.")
    parser.add_argument("wordkey", type=str, help="path to save word to label "
                        "mapping.")
    main(parser.parse_args())
