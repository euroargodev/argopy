#!/bin/bash

create_this_env () {
	# $1 is the name of the new environment
	# $2 is path to environment yaml file	
	
	# Overwrite name of the environment set in the file
	cp ${2} temp_env.yml
	var="name: $1"
	sed -i '' "1s/.*/$var/" temp_env.yml > /dev/null 2>&1
	
	# Eventually remove environment if exists:
	if conda env list | grep $1; then
	    printf "Replacing $1\n"
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

#create_this_env "argopy-tests-py37dev" "py3.7-dev.yml"
#create_this_env "argopy-tests-py38free-small" "py3.8-small-free.yml"
create_this_env "argopy-tests-py38free" "py3.8-free.yml"
#create_this_env "argopy-tests-py38dev" "py3.8-dev.yml"

exit 0