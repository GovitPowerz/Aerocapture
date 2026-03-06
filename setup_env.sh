rm -rf .venv
rm -rf .env
rm -rf test.db
uv sync
uv tree
cat <<EOF >>.venv/bin/activate

export PYTHONPATH="$PWD:$PWD/src/python"
EOF
