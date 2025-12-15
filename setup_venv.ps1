Write-Host "Creating .venv and installing dependencies..." -ForegroundColor Cyan

python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw "Failed to create venv" }

& .\.venv\Scripts\python -m pip install --upgrade pip
& .\.venv\Scripts\python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Run server:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\python app.py"


