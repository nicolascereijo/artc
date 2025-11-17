#Requires -Version 5.1
$ErrorActionPreference = "Stop"

Write-Host "--------------------------------------------------"
Write-Host "  ARtC Local Environment Installer (Python 3.12.x)"
Write-Host "--------------------------------------------------"

# 1) Determine project root (one level above this script)
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$PROJECT_DIR = Resolve-Path "$ScriptDir\.."
Set-Location $PROJECT_DIR

$PYTHON_VERSION = "3.12.7"
$PYTHON_ZIP = "python-$PYTHON_VERSION-embed-amd64.zip"
$PYTHON_URL = "https://www.python.org/ftp/python/$PYTHON_VERSION/$PYTHON_ZIP"

$PYTHON_LOCAL_DIR = Join-Path $PROJECT_DIR "python312"
$VENV_DIR = Join-Path $PROJECT_DIR ".artc"

# 2) Check for required tools
$Missing = @()

if (-not (Get-Command "curl.exe" -ErrorAction SilentlyContinue)) {
    if (-not (Get-Command "wget.exe" -ErrorAction SilentlyContinue)) {
        $Missing += "curl or wget"
    }
}

if ($Missing.Count -gt 0) {
    Write-Host "Error: missing required tools:"
    $Missing | ForEach-Object { Write-Host " - $_" }
    exit 1
}

# 3) Download Python embeddable package
if (-not (Test-Path $PYTHON_ZIP)) {
    Write-Host "Downloading Python $PYTHON_VERSION..."

    if (Get-Command "curl.exe" -ErrorAction SilentlyContinue) {
        curl.exe -L $PYTHON_URL -o $PYTHON_ZIP
    }
    elseif (Get-Command "wget.exe" -ErrorAction SilentlyContinue) {
        wget.exe $PYTHON_URL -O $PYTHON_ZIP
    }
}
else {
    Write-Host "Python zip already exists: $PYTHON_ZIP"
}

# 4) Extract Python locally
if (-not (Test-Path $PYTHON_LOCAL_DIR)) {
    Write-Host "Extracting Python..."
    Expand-Archive -Path $PYTHON_ZIP -DestinationPath $PYTHON_LOCAL_DIR
}
else {
    Write-Host "Local Python installation already exists: $PYTHON_LOCAL_DIR"
}

# Ensure python.exe exists
$PY_BIN = Join-Path $PYTHON_LOCAL_DIR "python.exe"

if (-not (Test-Path $PY_BIN)) {
    Write-Host "Error: python.exe not found in extracted folder"
    exit 1
}

# Workaround: embeddable Python needs pip & stdlib enabled
Write-Host "Installing pip into embeddable Python..."
& $PY_BIN -c "import ensurepip; ensurepip.bootstrap()"

# 5) Create virtual environment
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Creating virtual environment (.artc)..."
    & $PY_BIN -m venv $VENV_DIR
}
else {
    Write-Host "Virtual environment already exists: $VENV_DIR"
}

# 6) Activate environment
$Activate = Join-Path $VENV_DIR "Scripts\Activate.ps1"
. $Activate

Write-Host "Active environment: $(python --version)"

# 7) Install dependencies
if (Test-Path "requirements.txt") {
    Write-Host "Installing dependencies..."
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt
}
else {
    Write-Host "No requirements.txt file found. Skipping dependency installation."
}

# 8) Cleanup
Write-Host "Cleaning up build files..."
Remove-Item $PYTHON_ZIP -ErrorAction SilentlyContinue

# 9) Install local package
Write-Host "Installing local package..."

if (Test-Path "setup.py") {
    pip install .
}
else {
    Write-Host "Warning: no setup.py found. Skipping package installation."
}

# 10) Deactivate environment
deactivate
Write-Host "Environment deactivated."

# 11) Summary
$PY_INSTALLED_VER = & $PY_BIN --version
Write-Host "--------------------------------------------------"
Write-Host "Setup complete."
Write-Host "$PY_INSTALLED_VER installed in: $PYTHON_LOCAL_DIR"
Write-Host "Virtual environment created at: $VENV_DIR"
Write-Host ""
Write-Host "To activate the environment manually, run:"
Write-Host "  .\.artc\Scripts\Activate.ps1"
Write-Host "--------------------------------------------------"
