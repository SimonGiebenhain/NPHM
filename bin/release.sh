#!/bin/bash

while [[ "$#" -gt 0 ]]; do
	case $1 in
		-t|--test) test=1 ;;
		*) echo "Unknown parameter: $1"; exit 1 ;;
	esac
	shift
done

if [[ $test ]]; then
	twine_params="--repository testpypi"
else
	twine_params=""
fi

cd "${0%/*}/.."
rm -r dist/*
python -m build
twine upload $twine_params dist/*