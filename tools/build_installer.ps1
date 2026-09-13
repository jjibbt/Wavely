param(
    [Parameter(Mandatory = $true)][string]$PackageDir,
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$packageRoot = (Resolve-Path -LiteralPath $PackageDir).Path
if (-not $OutputDir) { $OutputDir = Join-Path $repoRoot 'build\installer' }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$outputRoot = (Resolve-Path -LiteralPath $OutputDir).Path

$expected = @(
    'Wavely.exe', 'README.txt',
    'runtime\WavelyVision.exe', 'runtime\WavelyEnrol.exe', 'runtime\WavelyTrain.exe',
    'config\haarcascade_profileface.xml', 'config\home_assistant_actions.json',
    'config\home_assistant_webhook_example.yaml',
    'config\wavely_app_settings.json', 'assets\wavely_icon.ico', 'assets\wavely_icon.png',
    'launchers\enrol_face_at_distance.bat', 'launchers\launch_wavely_terminal.bat',
    'launchers\train_face_model.bat'
)
$actual = @(Get-ChildItem -LiteralPath $packageRoot -File -Recurse | ForEach-Object {
    $_.FullName.Substring($packageRoot.Length + 1)
})
$unexpected = @(Compare-Object $expected $actual)
if ($unexpected.Count) { throw 'Package contents differ from the approved portable file list.' }
$actions = Get-Content -LiteralPath (Join-Path $packageRoot 'config\home_assistant_actions.json') -Raw | ConvertFrom-Json
if ($actions.webhook_url) { throw 'Portable package contains a webhook address.' }

$compiler = Get-Command ISCC.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if (-not $compiler) {
    $compiler = Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 7\ISCC.exe'
}
if (-not (Test-Path -LiteralPath $compiler)) { throw 'Inno Setup compiler was not found.' }
& $compiler "/DPackageDir=$packageRoot" "/O$outputRoot" (Join-Path $repoRoot 'installer\wavely.iss')
if ($LASTEXITCODE -ne 0) { throw 'Installer compilation failed.' }
Write-Output "Installer built in $outputRoot"
