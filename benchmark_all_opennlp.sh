LANGUAGES=$(ls data/*.sentences.train)

for train_file in $LANGUAGES
do
  lang=${train_file%.train}
  ./benchmark_language_opennlp.sh $lang $architecture
done
