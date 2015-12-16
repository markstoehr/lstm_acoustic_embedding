"""Collect a list of example words for performing detection experiments with.

Each word should have at least three speakers speaking the word.
"""

import argparse
import collections
import numpy
import os
import re

def word_stats(timitpath):
    utterance_pattern = re.compile("[^.]+")
    worddict = collections.defaultdict(list)
    for root, dirs, files in os.walk(timitpath):
        utterances = list(
            frozenset(utterance_pattern.findall(fl)[0] for fl in files))
        for utterance in utterances:
            words = "%s/%s.wrd" % (root, utterance)
            if not os.path.exists(words): continue
            for line in open(words, "r"):
                start, end, word = line.strip().split()
                worddict[word].append((root, utterance, int(start), int(end)))
    return worddict

def main(args):
    worddict = word_stats(args.timit)
    wordlist = open(args.wordlist, "w")
    instance_counts = [ (k, len(v)) for k, v in worddict.items()]
    counts = numpy.array([ (i, l) for i, (k, l) in enumerate(instance_counts)])
    word_indices = counts[counts[:, 1] >= 3, 0]
    for word_index in word_indices:
        word = instance_counts[word_index][0]
        for path, utterance, start, end in worddict[word]:
            wordlist.write( '%s %s/%s.wav %d %d\n' % (word,  path, utterance, start, end))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Assemble List of Words")
    parser.add_argument("timit", type=str, help="timit database location")
    parser.add_argument("wordlist", type=str, help="string for where to save "
                        "the word list")
    main(parser.parse_args())
