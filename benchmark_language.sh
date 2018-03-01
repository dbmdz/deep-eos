lang=$1
architecture=$2

function train {
  # Arguments:
  # $1 language
  # $2 architecture
  echo "Training for $lang with $2"
  python3 main.py train --training-file=$1.train --architecture=$2 --model-filename=${lang}_${architecture}.model
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
  python3 main.py tag --input-file $1.$2.modified --model-filename=${lang}_${architecture}.model > ${lang}_${architecture}_$2_output.txt
}

function final_evaluation {
  # Arguments:
  # $1 language
  # $2 data set name (usually dev or test)
  echo "Evaluation for $1 on $2 set"
  python3 eos-eval/eval.py -g $1.$2.gold -s ${lang}_${architecture}_$2_output.txt | tee ${lang}_${architecture}_$2_evaluation.txt
}

# Disable some TensorFlow warnings and information output
export TF_CPP_MIN_LOG_LEVEL=3

train $lang $architecture
prepare_dataset $lang dev
tagging $lang dev
final_evaluation $lang dev

prepare_dataset $lang test
tagging $lang test
final_evaluation $lang test

unset TF_CPP_MIN_LOG_LEVEL
