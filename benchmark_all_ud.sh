UD_FOLDER="ud-treebanks-v2.1"
ARCHITECTURES="lstm bi-lstm cnn"

for folder in $(ls $UD_FOLDER)
do
  current_path=$UD_FOLDER/$folder
  folder_files=$(ls $UD_FOLDER/$folder)

  train_file=$(echo $folder_files | tr ' ' '\n' | grep "\-ud\-train\.conllu" | grep -v "\.eos")

  dev_file=$(echo $folder_files | tr ' ' '\n' | grep "\-ud\-dev\.conllu" | grep -v "\.eos")

  test_file=$(echo $folder_files | tr ' ' '\n' | grep "\-ud\-test\.conllu" | grep -v "\.eos")

  if [[ ! -z "${train_file// }" ]] && [[ ! -z "${dev_file// }" ]] && \
  [[ ! -z "${test_file// }" ]]; then
    lang_prefix=$(echo $train_file | cut -d "-" -f 1)


    cat $current_path/$train_file | grep -P "^# text" | sed 's+# text = ++g' \
                                  | grep -P '[\.?!:;][^\n]?$' \
                                  > $current_path/$train_file.eos

    cat $current_path/$dev_file | grep -P "^# text" | sed 's+# text = ++g' \
                                | grep -P '[\.?!:;][^\n]?$' \
                                > $current_path/$dev_file.eos

    cat $current_path/$test_file | grep -P "^# text" | sed 's+# text = ++g' \
                                 | grep -P '[\.?!:;][^\n]?$' \
                                 > $current_path/$test_file.eos

    for architecture in $ARCHITECTURES
    do
      echo "Architecture: $architecture"
      ./benchmark_language_ud.sh $current_path/$train_file.eos \
                                 $current_path/$dev_file.eos \
                                 $current_path/$test_file.eos \
                                 $lang_prefix $architecture
    done
  fi
done
