SETIMES_LANG_PAIRS="en-hr en-mk en-ro en-sq en-sr en-tr bg-en bs-en el-en"
EUROPARL_LANG_PAIRS="en de"

BASE_SETIMES="SETIMES2"

BASE_EUROPARL="europarl-v7.de-en"

BASE_URL="http://schweter.eu/cloud/nn_eos"

function download {
  # Arguments:
  # $1 url
  # $2 desired path (path + filename + extension)
  pushd data
  if [ ! -s "${BASE_SETIMES}.$2" ]; then
    echo "Downloading corpus for $2"
    curl -LO $1.$2
  fi
  popd
}

function extract {
  # Arguments
  # $1 data set (compressed)
  pushd data
  if [ -s $1 ]; then
    echo "Extracting $1..."
    tar -xJf $1
  fi
  popd
}

for LANG_PAIR in $SETIMES_LANG_PAIRS
do
  download "${BASE_URL}/${BASE_SETIMES}" "${LANG_PAIR}.sentences.tar.xz"
  extract "${BASE_SETIMES}.${LANG_PAIR}.sentences.tar.xz"
done

for LANG_PAIR in $EUROPARL_LANG_PAIRS
do
  download "${BASE_URL}/${BASE_EUROPARL}" "${LANG_PAIR}.sentences.tar.xz"
  extract "${BASE_EUROPARL}.${LANG_PAIR}.sentences.tar.xz"
done
