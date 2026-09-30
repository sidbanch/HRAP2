# Installs HRAP as an app: right-click > Run with PowerShell, or run
#   powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/sidbanch/HRAP2/main/packaging/install_windows.ps1 | iex"
# Adds Desktop and Start menu shortcuts; the app keeps itself up to date from GitHub.
# Running it again repairs or reinstalls; motor files are never touched.
$ErrorActionPreference = "Stop"

$Repo = if ($env:HRAP_REPO) { $env:HRAP_REPO } else { "sidbanch/HRAP2" }
$Branch = if ($env:HRAP_BRANCH) { $env:HRAP_BRANCH } else { "main" }
$Root = Join-Path $env:LOCALAPPDATA "HRAP"

Write-Host "Installing HRAP from $Repo ($Branch)..."

# uv manages Python and packages; it installs per-user without admin rights.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
$Uv = (Get-Command uv).Source

New-Item -ItemType Directory -Force -Path $Root | Out-Null
$Py = Join-Path $Root "venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { & $Uv venv --python 3.12 (Join-Path $Root "venv") }

$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("hrap-" + [guid]::NewGuid())
New-Item -ItemType Directory -Path $Tmp | Out-Null
try {
    $Headers = @{ "User-Agent" = "HRAP-installer" }
    $Sha = (Invoke-RestMethod -Headers $Headers "https://api.github.com/repos/$Repo/commits/$Branch").sha
    Invoke-WebRequest -Headers $Headers "https://codeload.github.com/$Repo/zip/$Sha" -OutFile (Join-Path $Tmp "src.zip")
    Expand-Archive (Join-Path $Tmp "src.zip") -DestinationPath $Tmp
    $Src = (Get-ChildItem $Tmp -Directory | Select-Object -First 1).FullName
    Write-Host "Installing packages (the first time takes a minute)..."
    & $Py (Join-Path $Src "src\hrap\update.py") $Root $Repo $Branch --uv $Uv --source $Src --sha $Sha
    if ($LASTEXITCODE -ne 0) { throw "Package install failed." }
    Copy-Item (Join-Path $Src "src\hrap\resources\icon.ico") (Join-Path $Root "HRAP.ico") -Force
} finally {
    Remove-Item -Recurse -Force $Tmp
}

$Pyw = Join-Path $Root "venv\Scripts\pythonw.exe"
$Shell = New-Object -ComObject WScript.Shell
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
foreach ($Dir in @([Environment]::GetFolderPath("Desktop"), $StartMenu)) {
    $Link = $Shell.CreateShortcut((Join-Path $Dir "HRAP.lnk"))
    $Link.TargetPath = $Pyw
    $Link.Arguments = "-m hrap"
    $Link.WorkingDirectory = $Root
    $Link.IconLocation = Join-Path $Root "HRAP.ico"
    $Link.Save()
}

Write-Host "Done. HRAP is on your Desktop and in the Start menu."
Start-Process $Pyw -ArgumentList "-m hrap" -WorkingDirectory $Root
