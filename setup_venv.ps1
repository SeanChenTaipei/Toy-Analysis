param(
    [string]$VenvPath = ".venv_ready",
    [string]$PythonExe = "python",
    [switch]$TryInstallFromRequirements = $true
)

$ErrorActionPreference = "Stop"

function Run-Command {
    param(
        [string]$Message,
        [scriptblock]$Action,
        [switch]$Optional
    )

    Write-Output $Message
    & $Action
    if ($LASTEXITCODE -ne 0) {
        if ($Optional) {
            Write-Warning "Optional step failed (exit=${LASTEXITCODE}): $Message"
        }
        else {
            throw "Step failed (exit=${LASTEXITCODE}): $Message"
        }
    }
}

if (-not (Test-Path $VenvPath)) {
    Run-Command "[1/4] Creating venv: $VenvPath (system-site-packages)" {
        & $PythonExe -m venv $VenvPath --without-pip --system-site-packages
    }
}
else {
    Write-Output "[1/4] Reusing existing venv: $VenvPath"
}

$venvPython = Join-Path $VenvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Venv python not found: $venvPython"
}

Run-Command "[2/4] Verifying required imports" {
    & $venvPython -c "import numpy,pandas,matplotlib,seaborn,scipy,statsmodels; print('core imports ok')"
}

if ($TryInstallFromRequirements) {
    Run-Command "[3/4] Optional: install/upgrade packages from requirements.txt" {
        & $PythonExe -m pip --python $venvPython install -r requirements.txt
    } -Optional
}
else {
    Write-Output "[3/4] Skipped requirements install by flag"
}

Run-Command "[4/4] Optional: register ipykernel" {
    & $venvPython -m ipykernel install --user --name "toy-analysis-venv" --display-name "Python (toy-analysis-venv)"
} -Optional

Write-Output "Done. Activate with: .\$VenvPath\Scripts\Activate.ps1"
