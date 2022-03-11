#!/usr/bin/env bash

create_this_env () {
	# $1 is the name of the new environment
	# $2 is path to environment yaml file	
	
	# Overwrite name of the environment set in the file
	cp ${2} temp_env.yml
	var="name: $1"
	sed -i '' "1s/.*/$var/" temp_env.yml > /dev/null 2>&1
	
	# Eventually remove environment if exists:
	if conda env list | grep $1; then
	    printf "$1 already exists, remove and re-create this environment...\n"
		# conda remove --quiet --name $1 --all --yes --json > tmp.json
		  conda remove --quiet --name $1 --all --yes --json > /dev/null 2>&1
	else
		  printf "Creating conda environment $1 ...\n"
	fi

	# Create the environment from file
	# conda env create --quiet --file temp_env.yml --json >> tmp.json
	conda env create --quiet --file temp_env.yml --json > /dev/null 2>&1
	
	# Clean-up
	/bin/rm tmp.json > /dev/null 2>&1
	/bin/rm temp_env.yml > /dev/null 2>&1


}

add_to_ipykernel () {
  # Possibly add it the Jupyter kernels:
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate ${1}
   echo ${1}
   python -m ipykernel install --user --name=$1
}


POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
#    -n|--name)
#      NAME="$2"
#      shift # past argument
#      shift # past value
#      ;;
#    -f|--file)
#      FILE="$2"
#      shift # past argument
#      shift # past value
#      ;;
    -a|--all)
      declare -A ENV_LIST
      ENV_LIST[argopy-tests-py37dev]="py3.7-dev.yml"
      ENV_LIST[argopy-tests-py37free]="py3.7-free.yml"
      ENV_LIST[argopy-tests-py38dev]="py3.8-dev.yml"
      ENV_LIST[argopy-tests-py38free]="py3.8-free.yml"
      ENV_LIST[argopy-tests-py38free-small]="py3.8-small-free.yml"
      shift # past argument
      ;;
    --py37dev)
      declare -A ENV_LIST
      ENV_LIST[argopy-tests-py37dev]="py3.7-dev.yml"
      shift # past argument
      ;;
    --py37free)
      declare -A ENV_LIST
      ENV_LIST[argopy-tests-py37free]="py3.7-free.yml"
      shift # past argument
      ;;
    --py38dev)
      declare -A ENV_LIST
      ENV_LIST[argopy-tests-py38dev]="py3.8-dev.yml"
      shift # past argument
      ;;
    --py38free)
      declare -A ENV_LIST
      ENV_LIST[argopy-tests-py38free]="py3.8-free.yml"
      shift # past argument
      ;;
    --py38small)
      declare -A ENV_LIST
      ENV_LIST[argopy-tests-py38free-small]="py3.8-small-free.yml"
      shift # past argument
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

for NAME in "${!ENV_LIST[@]}"
do
   :
   FILE=${ENV_LIST[$NAME]}
   create_this_env ${NAME} ${FILE}
   add_to_ipykernel ${NAME}
done

exit 0