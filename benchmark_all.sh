ARCHITECTURES="lstm"

for architecture in $ARCHITECTURES
do
  ./benchmark_language.sh $architecture
done
