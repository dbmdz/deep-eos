LANGUAGES=$(ls data/*.sentences.train)
ARCHITECTURES="lstm bi-lstm cnn"

for architecture in $ARCHITECTURES
do
  for train_file in $LANGUAGES
  do
    lang=${train_file%.train}
    ./benchmark_language.sh $lang $architecture
  done
done
