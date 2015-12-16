# get data
TIMIT=~/Data/timit

if [ -z $1 ] ; then
    basedir=`pwd`
else
    basedir=$1
fi
echo "Building directory in $basedir"

datadir=$basedir/data
mkdir -p $datadir

# get word list
# python data_preparation/collect_timit_words.py $TIMIT/test $datadir/test_wordlist
# python data_preparation/collect_timit_words.py $TIMIT/train $datadir/train_wordlist

# generate MFCC features
echo "Generating testing MFCC features"
python data_preparation/wordlist_to_mfccs.py $datadir/test_wordlist $datadir/test_mfccs.hdf5 $datadir/test_wordkey.txt
echo "Generating training MFCC features"
python data_preparation/wordlist_to_mfccs.py $datadir/train_wordlist $datadir/train_mfccs.hdf5 $datadir/train_wordkey.txt
