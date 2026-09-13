$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$buildRoot = if ($env:WAVELY_BUILD_DIR) { $env:WAVELY_BUILD_DIR } else { Join-Path $repoRoot 'build' }
$python = Join-Path $buildRoot 'venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    py -3.11 -m venv (Join-Path $buildRoot 'venv')
}
& $python -m pip install --disable-pip-version-check -r (Join-Path $repoRoot 'requirements-build.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }

$dist = Join-Path $buildRoot 'dist'
$work = Join-Path $buildRoot 'work'
$spec = Join-Path $buildRoot 'spec'
foreach ($target in @(
    @{ Name = 'Wavely'; Script = 'wavely_dashboard.py'; Mode = '--windowed'; Media = $false },
    @{ Name = 'WavelyVision'; Script = 'wavely_vision.py'; Mode = '--console'; Media = $true },
    @{ Name = 'WavelyEnrol'; Script = 'video_enrol.py'; Mode = '--console'; Media = $true },
    @{ Name = 'WavelyTrain'; Script = 'train_all_faces.py'; Mode = '--console'; Media = $false }
)) {
    $pyInstallerArgs = @('-m', 'PyInstaller', '--noconfirm', '--onefile', $target.Mode,
        '--name', $target.Name, '--paths', (Join-Path $repoRoot 'src'),
        '--icon', (Join-Path $repoRoot 'assets\wavely_icon.ico'),
        '--distpath', $dist, '--workpath', $work, '--specpath', $spec)
    if ($target.Name -eq 'Wavely') { $pyInstallerArgs += @('--hidden-import', 'comtypes.client', '--collect-submodules', 'comtypes') }
    if ($target.Media) { $pyInstallerArgs += @('--collect-data', 'mediapipe') }
    if ($target.Name -in @('WavelyEnrol', 'WavelyTrain')) { $pyInstallerArgs += @('--collect-data', 'cv2') }
    $pyInstallerArgs += Join-Path $repoRoot ('src\' + $target.Script)
    & $python @pyInstallerArgs
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $($target.Name)" }
}

$package = Join-Path $buildRoot 'package'
if (Test-Path -LiteralPath $package) {
    Remove-Item -LiteralPath $package -Recurse -Force
}
New-Item -ItemType Directory -Path (Join-Path $package 'runtime') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $package 'config') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $package 'assets') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $package 'launchers') -Force | Out-Null

Copy-Item (Join-Path $dist 'Wavely.exe') (Join-Path $package 'Wavely.exe')
foreach ($runtimeName in @('WavelyVision.exe', 'WavelyEnrol.exe', 'WavelyTrain.exe')) {
    Copy-Item (Join-Path $dist $runtimeName) (Join-Path $package "runtime\$runtimeName")
}
Copy-Item (Join-Path $repoRoot 'README.txt') (Join-Path $package 'README.txt')
Copy-Item (Join-Path $repoRoot 'config\haarcascade_profileface.xml') (Join-Path $package 'config\haarcascade_profileface.xml')
Copy-Item (Join-Path $repoRoot 'config\wavely_app_settings.json') (Join-Path $package 'config\wavely_app_settings.json')
Copy-Item (Join-Path $repoRoot 'config\home_assistant_webhook_example.yaml') (Join-Path $package 'config\home_assistant_webhook_example.yaml')
Copy-Item (Join-Path $repoRoot 'assets\wavely_icon.ico') (Join-Path $package 'assets\wavely_icon.ico')
Copy-Item (Join-Path $repoRoot 'assets\wavely_icon.png') (Join-Path $package 'assets\wavely_icon.png')
foreach ($launcher in @('enrol_face_at_distance.bat', 'launch_wavely_terminal.bat', 'train_face_model.bat')) {
    Copy-Item (Join-Path $repoRoot "launchers\$launcher") (Join-Path $package "launchers\$launcher")
}

$actions = Get-Content -LiteralPath (Join-Path $repoRoot 'config\home_assistant_actions.json') -Raw | ConvertFrom-Json
$actions.webhook_url = ''
$actions | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $package 'config\home_assistant_actions.json') -Encoding utf8

Write-Output "Executables built in $dist"
Write-Output "Installer package staged in $package"
