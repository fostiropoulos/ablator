
PY_VERSION=$(python --version | grep -Eo '[0-9]\.[0-9]+\.[0-9]+')
CONTAINER_NAME="ablator"
ARGS=("${@}")
for ((i=0;i<$#;i++))
do
   case ${ARGS[$i]} in
       # -- option
       --py-3-10-12 )
       PY_VERSION="3.10.12";;
       * ) CONTAINER_NAME=${ARGS[$i]};;
   esac

done

echo "${PY_VERSION}"

docker build --build-arg="PY_VERSION=${PY_VERSION}" --tag ${CONTAINER_NAME} .

