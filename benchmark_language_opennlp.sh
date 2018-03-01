lang=$1
architecture=$2

# Set JAVA_HOME here when needed
#export JAVA_HOME="/home/stefan/Downloads/jdk1.8.0_112"

# Path to opennlp "binary", which is actually a shell script, where you need to
# adjust the -Xmx value when training on Europarl. A recommended value is:
# -Xmx4096m
OPENNLP="opennlp"

function train {
  # Arguments:
  # $1 language
  # $2 architecture
  echo "Training for $lang with $2"
  ${OPENNLP} SentenceDetectorTrainer -model own.bin -lang $1 -data $1.train -encoding UTF-8
}

function prepare_dataset {
  # Arguments:
  # $1 language
  # $2 data set name (usually dev or test)
  echo "Prepare $2 set for $1"
  cat $lang.$2 | tr '\n' ' ' > $lang.$2.modified
  cat $lang.$2 | awk '{ print $0 "</eos>" }' > $lang.$2.gold
}

function tagging {
  # Arguments:
  # $1 language
  # $2 data set name (usually dev or test)
  echo "Tagging $2 set for $1"
  ${OPENNLP} SentenceDetector own.bin < $1.$2.modified > ${lang}_${architecture}_$2_output_temp.txt
  cat ${lang}_${architecture}_$2_output_temp.txt | sed '$d' | awk '{ print $0 "</eos>" }' > ${lang}_${architecture}_$2_output.txt
}

function final_evaluation {
  # Arguments:
  # $1 language
  # $2 data set name (usually dev or test)
  echo "Evaluation for $1 on $2 set"
  python3 eos-eval/eval.py -g $1.$2.gold -s ${lang}_${architecture}_$2_output.txt | tee ${lang}_${architecture}_$2_evaluation.txt
}

train $lang $architecture
prepare_dataset $lang dev
tagging $lang dev
final_evaluation $lang dev

prepare_dataset $lang test
tagging $lang test
final_evaluation $lang test
