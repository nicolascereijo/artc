#!/usr/bin/env bash
set -e

echo "--------------------------------------------------"
echo "  ARtC Local Environment Installer (Python 3.12.x)"
echo "--------------------------------------------------"

# 1) Determine project root (where this script is located)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_VERSION="3.12.7"
PYTHON_TARBALL="Python-${PYTHON_VERSION}.tgz"
PYTHON_SRC_DIR="Python-${PYTHON_VERSION}"
PYTHON_LOCAL_DIR="${PROJECT_DIR}/python312"
VENV_DIR="${PROJECT_DIR}/.artc"

# 2) Check for required build tools
MISSING_TOOLS=()

for tool in tar make gcc; do
    if ! command -v "$tool" &>/dev/null; then
        MISSING_TOOLS+=("$tool")
    fi
done

# At least one of curl or wget must exist
if ! command -v wget &>/dev/null && ! command -v curl &>/dev/null; then
    MISSING_TOOLS+=("wget or curl")
fi

if [ ${#MISSING_TOOLS[@]} -ne 0 ]; then
    echo "Error: the following required tools are missing:"
    printf ' - %s\n' "${MISSING_TOOLS[@]}"
    echo "Please install them and rerun this script."
    exit 1
fi

# 3) Download Python source if not already present
if [ ! -f "$PYTHON_TARBALL" ]; then
    echo "Downloading Python ${PYTHON_VERSION}..."
    if command -v wget &>/dev/null; then
        wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TARBALL}"
    elif command -v curl &>/dev/null; then
        curl -O "https://www.python.org/ftp/python/${PYTHON_VERSION}/${PYTHON_TARBALL}"
    else
        echo "Error: neither 'wget' nor 'curl' found. Cannot download Python source."
        exit 1
    fi
else
    echo "Python source archive already exists: ${PYTHON_TARBALL}"
fi

# 4) Extract and build locally
if [ ! -d "$PYTHON_SRC_DIR" ]; then
    tar -xf "$PYTHON_TARBALL"
fi

cd "$PYTHON_SRC_DIR"

if [ ! -d "$PYTHON_LOCAL_DIR" ]; then
    echo "Building and installing Python ${PYTHON_VERSION} locally..."
    ./configure --prefix="$PYTHON_LOCAL_DIR" --enable-optimizations
    make -j
    make install
else
    echo "Local Python installation already exists: ${PYTHON_LOCAL_DIR}"
fi

# 5) Create virtual environment inside project
cd "$PROJECT_DIR"

PY_BIN="${PYTHON_LOCAL_DIR}/bin/python3.12"
if [ ! -x "$PY_BIN" ]; then
    echo "Error: Local Python binary not found at $PY_BIN"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment (.artc)..."
    "$PY_BIN" -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists: $VENV_DIR"
fi

# 6) Activate the environment
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Active environment: $(python --version)"

# 7) Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
else
    echo "No requirements.txt file found. Skipping dependency installation."
fi

# 8) Optional cleanup (commented out)
echo "Cleaning up build files..."
rm -rf "$PYTHON_SRC_DIR" "$PYTHON_TARBALL"

# 9) Show final summary
PY_INSTALLED_VER=$("$PY_BIN" --version | awk '{print $2}')
echo "--------------------------------------------------"
echo "Setup complete."
echo "Python ${PY_INSTALLED_VER} installed in: $PYTHON_LOCAL_DIR"
echo "Virtual environment created at: $VENV_DIR"
echo
echo "To activate the environment manually, run:"
echo "  source .artc/bin/activate"
echo "--------------------------------------------------"
