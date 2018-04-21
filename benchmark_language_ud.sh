train_file=$1
devel_file=$2
test_file=$3
lang_prefix=$4
architecture=$5

function train {
  echo "Training on ${train_file} for ${architecture}"
  python3 main.py train --training-file=${train_file} \
                        --architecture=${architecture} \
                        --model-filename=${lang_prefix}_${architecture}.model \
                        --vocab-filename=${lang_prefix}_${architecture}.vocab
}

function prepare_dataset {
  # Arguments:
  # $1 dataset filename
  echo "Prepare $1 set"
  cat $1 | tr '\n' ' ' > $1.modified
  cat $1 | awk '{ print $0 "</eos>" }' > $1.gold
}

function tagging {
  # Arguments:
  # $1 dataset filename
  # $2 unique dataset name
  echo "Tagging $1"
  python3 main.py tag --input-file $1.modified \
                      --model-filename=${lang_prefix}_${architecture}.model \
                      --vocab-filename=${lang_prefix}_${architecture}.vocab \
                      > ${lang_prefix}_${architecture}_$2_output.txt
}

function final_evaluation {
  # Arguments:
  # $1 dataset filename
  # $2 unique dataset name
  echo "Evaluation on $1 set"
  python3 eos-eval/eval.py -g $1.gold \
                           -s ${lang_prefix}_${architecture}_$2_output.txt \
                           | tee ${lang_prefix}_${architecture}_$2_evaluation.txt
}

# Disable some TensorFlow warnings and information output
export TF_CPP_MIN_LOG_LEVEL=3

train
prepare_dataset $devel_file "dev"
tagging $devel_file "dev"
final_evaluation $devel_file "dev"

prepare_dataset $test_file "test"
tagging $test_file "test"
final_evaluation $test_file "test"

unset TF_CPP_MIN_LOG_LEVEL
