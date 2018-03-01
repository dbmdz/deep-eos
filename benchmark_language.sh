architecture=$1

function train {
  # Arguments:
  # $1 training file
  echo "Training with ${architecture} architecture"
  python3 main.py train --training-file=$1 --architecture=${architecture}
}

function prepare_dataset {
  # Arguments:
  # $1 data set name (usually dev or test)
  echo "Prepare $1 set"
  cat $1 | tr '\n' ' ' > $1.modified
  cat $1 | awk '{ print $0 "</eos>" }' > $1.gold
}

function tagging {
  # Arguments:
  # $1 data set name (usually dev or test)
  echo "Tagging $1 set"
  python3 main.py tag --input-file $1.modified > $1_${architecture}_output.txt
}

function final_evaluation {
  # Arguments:
  # $1 data set name (usually dev or test)
  echo "Evaluation on $1 set"
  python3 eos-eval/eval.py -g $1.gold -s $1_${architecture}_output.txt | tee $1_${architecture}_evaluation.txt
}

# Disable some TensorFlow warnings and information output
export TF_CPP_MIN_LOG_LEVEL=3

train ./data/train
prepare_dataset ./data/dev
tagging ./data/dev
final_evaluation ./data/dev

prepare_dataset ./data/test
tagging ./data/test
final_evaluation ./data/test

unset TF_CPP_MIN_LOG_LEVEL
