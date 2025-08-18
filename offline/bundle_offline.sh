#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

WHEELHOUSE="${PWD}/wheelhouse"
mkdir -p "$WHEELHOUSE"

# Build our package wheel
pushd ../packages/spintransport
python -m build --wheel
cp dist/*.whl "$WHEELHOUSE/"
popd

# Collect third-party deps (pin/adjust si deseas)
REQS="$(cat <<'EOF'
numpy
scipy
matplotlib
imageio>=2.34.0
tqdm
EOF
)"
echo "$REQS" > requirements.txt

# Download dependencies into wheelhouse (no install)
pip download -r requirements.txt -d "$WHEELHOUSE"

echo
echo "Offline wheelhouse listo en: $WHEELHOUSE"
echo "Instalación offline:"
echo "  python3 -m venv .venv && source .venv/bin/activate"
echo "  pip install --no-index --find-links wheelhouse spintransport"
