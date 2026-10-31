[CmdletBinding()]
Param(
    [Parameter()][Alias('i')][switch]$Install,
    [Parameter()][Alias('h')][switch]$Help,
    [Parameter()][Alias('a')][string]$Architecture,
    [Parameter()][string]$Name,
    [Parameter()][switch]$Sim,
    [Parameter()][switch]$RustTools,
    [Parameter()][switch]$DryRun
)

if ($DryRun) {
    $rustToolFeatures = if ($RustTools) { "rust-tools" } else { "none" }
    if ($Sim) {
        $simFeatures = if ($RustTools) { "sim,rocm,directml,rust-tools" } else { "sim,rocm,directml" }
        Write-Output "mode=sim packages=zed,cli,sim_worker,auto_update_helper zed_features=$simFeatures remote_features=$rustToolFeatures worker_features=rocm,directml include_sim_worker=true rust_tools=$($RustTools.ToString().ToLower())"
    }
    else {
        Write-Output "mode=default packages=zed,cli,auto_update_helper zed_features=$rustToolFeatures remote_features=$rustToolFeatures include_sim_worker=false rust_tools=$($RustTools.ToString().ToLower())"
    }
    exit 0
}

. "$PSScriptRoot/lib/workspace.ps1"

# https://stackoverflow.com/questions/57949031/powershell-script-stops-if-program-fails-like-bash-set-o-errexit
Set-PSDebug -Off
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$buildSuccess = $false
$canCodeSign = $false

$requiredProductVariables = @(
    'ZED_PRODUCT_ID', 'ZED_PRODUCT_DISPLAY_NAME', 'ZED_PRODUCT_EXECUTABLE',
    'ZED_PRODUCT_BUNDLE_ID', 'ZED_PRODUCT_URL_SCHEME', 'ZED_PRODUCT_DATA_NAMESPACE', 'ZED_PRODUCT_ICON_SET',
    'ZED_PRODUCT_APP_FEATURES', 'ZED_PRODUCT_REMOTE_FEATURES',
    'ZED_PRODUCT_WINDOWS_INSTALLER_ID', 'ZED_PRODUCT_ARTIFACT_NAME'
)
foreach ($variableName in $requiredProductVariables) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($variableName))) {
        throw "Missing resolved product variable: $variableName. Run cargo xtask bundle --product <id>."
    }
}

$OSArchitecture = switch ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture) {
    "X64" { "x86_64" }
    "Arm64" { "aarch64" }
    default { throw "Unsupported architecture" }
}

$Architecture = if ($Architecture) {
    $Architecture
} else {
    $OSArchitecture
}

$CargoTargetDir = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { "./target" }
$CargoOutDir = "$CargoTargetDir/$Architecture-pc-windows-msvc/release"

function Get-VSArch {
    param(
        [string]$Arch
    )

    switch ($Arch) {
        "x86_64" { "amd64" }
        "aarch64" { "arm64" }
    }
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere -PathType Leaf)) {
    throw "Visual Studio Installer discovery tool was not found at $vswhere"
}
$visualStudioPath = & $vswhere -latest -products * -property installationPath
$visualStudioShell = Join-Path $visualStudioPath "Common7\Tools\Launch-VsDevShell.ps1"
if (-not (Test-Path $visualStudioShell -PathType Leaf)) {
    throw "Visual Studio developer shell was not found at $visualStudioShell"
}
$visualStudioCmake = Join-Path $visualStudioPath "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if (-not (Test-Path $visualStudioCmake -PathType Leaf)) {
    throw "Visual Studio CMake was not found at $visualStudioCmake"
}
# wasmtime-c-api-impl must use a native Windows CMake. An MSYS CMake from Git's
# PATH interprets absolute Windows output paths incorrectly and leaves wasm.h missing.
$env:CMAKE = $visualStudioCmake
Push-Location
& $visualStudioShell -Arch (Get-VSArch -Arch $Architecture) -HostArch (Get-VSArch -Arch $OSArchitecture)
Pop-Location

$target = "$Architecture-pc-windows-msvc"

if ($Help) {
    Write-Output "Usage: test.ps1 [-Install] [-Help]"
    Write-Output "Build the installer for Windows.\n"
    Write-Output "Options:"
    Write-Output "  -Architecture, -a Which architecture to build (x86_64 or aarch64)"
    Write-Output "  -Install, -i      Run the installer after building."
    Write-Output "  -Sim             Include Sim, accelerator backends, worker, and assets."
    Write-Output "  -RustTools         Include the Cargo tool window and matching remote-server support."
    Write-Output "  -DryRun            Print the selected package plan without building it."
    Write-Output "  -Help, -h         Show this help message."
    exit 0
}

$channel = if ($env:ZED_RELEASE_CHANNEL) {
    $env:ZED_RELEASE_CHANNEL
} else {
    Get-Content "crates/zed/RELEASE_CHANNEL"
}
$env:ZED_RELEASE_CHANNEL = $channel
$env:RELEASE_CHANNEL = $channel
$bundlePhase = if ($env:ZED_BUNDLE_PHASE) { $env:ZED_BUNDLE_PHASE } else { "all" }
if (@('all', 'application', 'remote-server', 'package') -notcontains $bundlePhase) {
    throw "Unsupported bundle phase: $bundlePhase"
}

function CheckEnvironmentVariables {
    if($env:CI) {
        $requiredVars = @('ZED_WORKSPACE', 'RELEASE_VERSION', 'ZED_RELEASE_CHANNEL')

        foreach ($var in $requiredVars) {
            if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($var))) {
                Write-Error "$var is not set"
                exit 1
            }
        }
    }

    $signingMode = python "$PSScriptRoot/check-signing-environment" --platform windows
    if ($LASTEXITCODE -ne 0) { throw "Signing environment validation failed" }
    $script:canCodeSign = $signingMode -eq 'production'
    if ($script:canCodeSign -and -not (Get-Module -ListAvailable -Name TrustedSigning | Where-Object { $_.Version -eq [Version]'0.5.8' })) {
        Install-Module -Name TrustedSigning -RequiredVersion 0.5.8 -Scope CurrentUser -Repository PSGallery -Force -ErrorAction Stop
    }

}

function PrepareForBundle {
    if (Test-Path "$innoDir") {
        Remove-Item -Path "$innoDir" -Recurse -Force
    }
    New-Item -Path "$innoDir" -ItemType Directory -Force
    Copy-Item -Path "$env:ZED_WORKSPACE\crates\zed\resources\windows\*" -Destination "$innoDir" -Recurse -Force
    New-Item -Path "$innoDir\make_appx" -ItemType Directory -Force
    New-Item -Path "$innoDir\appx" -ItemType Directory -Force
    New-Item -Path "$innoDir\bin" -ItemType Directory -Force
    New-Item -Path "$innoDir\tools" -ItemType Directory -Force

    if ($bundlePhase -ne 'package') {
        rustup target add $target
    }
}

function GenerateLicenses {
    if ($env:ZED_PREGENERATED_LICENSES) {
        if (-not (Test-Path "assets/licenses.md" -PathType Leaf)) {
            throw "The pre-generated license artifact is missing"
        }
    }
    else {
        . $PSScriptRoot/generate-licenses.ps1
    }
}

function BuildProductBinaries {
    Write-Output "Building product binaries for channel: $channel"
    if ($Sim) {
        $features = (($env:ZED_PRODUCT_APP_FEATURES -split "," | ForEach-Object { "zed/$_" }) -join ",") + ",zed/sim,zed/rocm,sim_worker/rocm,zed/directml,sim_worker/directml"
        if ($RustTools) {
            $features = "$features,zed/rust-tools"
        }
        cargo build --release --package zed --package cli --package sim_worker --package auto_update_helper --no-default-features --features $features --target $target
    }
    else {
        $applicationFeatures = (($env:ZED_PRODUCT_APP_FEATURES -split ',') | ForEach-Object { "zed/$_" }) -join ','
        cargo build --release --package zed --package cli --package auto_update_helper --no-default-features --features $applicationFeatures --target $target
    }
    foreach ($binary in @('zed.exe', 'cli.exe', 'auto_update_helper.exe')) {
        $metadata = (Get-Item "$CargoOutDir\$binary").VersionInfo
        if ($metadata.ProductName -ne $env:ZED_PRODUCT_DISPLAY_NAME -or $metadata.FileVersion -ne $env:RELEASE_VERSION -or $metadata.ProductVersion -ne $env:RELEASE_VERSION) {
            throw "Product metadata mismatch in ${binary}: $($metadata.ProductName) $($metadata.FileVersion)"
        }
    }
}

function StageProductBinaries {
    Copy-Item -Path "$CargoOutDir\zed.exe" -Destination "$innoDir\$env:ZED_PRODUCT_EXECUTABLE.exe" -Force
    Copy-Item -Path "$CargoOutDir\cli.exe" -Destination "$innoDir\cli.exe" -Force
    if ($Sim) {
        Copy-Item -Path ".\$CargoOutDir\sim-worker.exe" -Destination "$innoDir\sim-worker.exe" -Force
    }
    Copy-Item -Path "$CargoOutDir\auto_update_helper.exe" -Destination "$innoDir\auto_update_helper.exe" -Force
}

function BuildRemoteServer {
    Write-Output "Building remote_server for $target"
    if (-not [string]::IsNullOrWhiteSpace($env:ZED_PRODUCT_REMOTE_FEATURES)) {
        cargo build --release --package remote_server --no-default-features --features $env:ZED_PRODUCT_REMOTE_FEATURES --target $target
    }
    else {
        cargo build --release --package remote_server --no-default-features --target $target
    }

}

function PackageRemoteServer {
    $remoteServerSrc = (Resolve-Path "$CargoOutDir\remote_server.exe").Path

    if ($canCodeSign) {
        Write-Output "Code signing remote_server.exe"
        & "$innoDir\sign.ps1" $remoteServerSrc
    }

    $remoteServerDst = "$CargoTargetDir\$env:ZED_PRODUCT_ID-remote-server-windows-$Architecture.zip"
    Write-Output "Compressing remote_server to $remoteServerDst"
    Compress-Archive -Path $remoteServerSrc -DestinationPath $remoteServerDst -Force

    Write-Output "Remote server compressed successfully"
}

function ZipProductDebugSymbols {
    $items = @(
        "$CargoOutDir\zed.pdb",
        "$CargoOutDir\cli.pdb",
        "$CargoOutDir\auto_update_helper.pdb",
        "$CargoOutDir\remote_server.pdb"
    )
    if ($Sim) {
        $items += ".\$CargoOutDir\sim-worker.pdb"
    }

    Compress-Archive -Path $items -DestinationPath $debugArchive -Force
}


function SignZedAndItsFriends {
    if (-not $canCodeSign) {
        return
    }

    $files = @(Get-ChildItem -LiteralPath $innoDir -Recurse -File | Where-Object { $_.Extension -in '.exe', '.dll' })
    if ($files.Count -eq 0) { throw "No staged native payloads were found for signing" }
    foreach ($file in $files) {
        & "$innoDir\sign.ps1" $file.FullName
    }

}

function DownloadAMDGpuServices {
    # If you update the AGS SDK version, please also update the version in `crates/gpui/src/platform/windows/directx_renderer.rs`
    $url = "https://codeload.github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/zip/refs/tags/v6.3.0"
    $zipPath = ".\AGS_SDK_v6.3.0.zip"
    # Download the AGS SDK zip file
    Invoke-WebRequest -Uri $url -OutFile $zipPath
    # Extract the AGS SDK zip file
    Expand-Archive -Path $zipPath -DestinationPath "." -Force
}

function DownloadConpty {
    $url = "https://github.com/microsoft/terminal/releases/download/v1.23.13503.0/Microsoft.Windows.Console.ConPTY.1.23.251216003.nupkg"
    $zipPath = ".\Microsoft.Windows.Console.ConPTY.1.23.251216003.nupkg"
    Invoke-WebRequest -Uri $url -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath ".\conpty" -Force
}

function CollectFiles {
    Move-Item -Path "$innoDir\cli.exe" -Destination "$innoDir\bin\$env:ZED_PRODUCT_EXECUTABLE.exe" -Force
    Move-Item -Path "$innoDir\zed.sh" -Destination "$innoDir\bin\$env:ZED_PRODUCT_EXECUTABLE" -Force
    Move-Item -Path "$innoDir\auto_update_helper.exe" -Destination "$innoDir\tools\auto_update_helper.exe" -Force
    if($Architecture -eq "aarch64") {
        New-Item -Type Directory -Path "$innoDir\arm64" -Force
        Move-Item -Path ".\conpty\build\native\runtimes\arm64\OpenConsole.exe" -Destination "$innoDir\arm64\OpenConsole.exe" -Force
        Move-Item -Path ".\conpty\runtimes\win-arm64\native\conpty.dll" -Destination "$innoDir\conpty.dll" -Force
    }
    else {
        New-Item -Type Directory -Path "$innoDir\x64" -Force
        New-Item -Type Directory -Path "$innoDir\arm64" -Force
        Move-Item -Path ".\AGS_SDK-6.3.0\ags_lib\lib\amd_ags_x64.dll" -Destination "$innoDir\amd_ags_x64.dll" -Force
        Move-Item -Path ".\conpty\build\native\runtimes\x64\OpenConsole.exe" -Destination "$innoDir\x64\OpenConsole.exe" -Force
        Move-Item -Path ".\conpty\build\native\runtimes\arm64\OpenConsole.exe" -Destination "$innoDir\arm64\OpenConsole.exe" -Force
        Move-Item -Path ".\conpty\runtimes\win-x64\native\conpty.dll" -Destination "$innoDir\conpty.dll" -Force
    }
}

function BuildInstaller {
    $issFilePath = "$innoDir\zed.iss"
    if (@('stable', 'preview', 'nightly', 'dev') -notcontains $channel) {
        throw "Can't bundle installer for unsupported channel $channel."
    }
    $appId = $env:ZED_PRODUCT_WINDOWS_INSTALLER_ID
    $appIconSource = Join-Path $env:ZED_PRODUCT_ICON_SET "app-icon.ico"
    if (-not (Test-Path $appIconSource -PathType Leaf)) {
        throw "Product installer icon does not exist: $appIconSource"
    }
    $appName = $env:ZED_PRODUCT_DISPLAY_NAME
    $appDisplayName = $appName
    $appSetupName = [System.IO.Path]::GetFileNameWithoutExtension($env:ZED_PRODUCT_ARTIFACT_NAME)
    $appMutex = "$env:ZED_PRODUCT_ID-$channel-Instance-Mutex"
    $appExeName = $env:ZED_PRODUCT_EXECUTABLE
    $regValueName = $env:ZED_PRODUCT_DATA_NAMESPACE
    $appUserId = $env:ZED_PRODUCT_BUNDLE_ID
    $appShellNameShort = "&$env:ZED_PRODUCT_DISPLAY_NAME"

    # Windows runner 2022 default has iscc in PATH, https://github.com/actions/runner-images/blob/main/images/windows/Windows2022-Readme.md
    # Currently, we are using Windows 2022 runner.
    # Windows runner 2025 doesn't have iscc in PATH for now, https://github.com/actions/runner-images/issues/11228
    $innoSetupPath = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"

    $definitions = @{
        "AppId"          = $appId
        "ProductIconFile" = $appIconSource
        "OutputDir"      = "$CargoTargetDir\release"
        "AppSetupName"   = $appSetupName
        "AppName"        = $appName
        "AppDisplayName" = $appDisplayName
        "RegValueName"   = $regValueName
        "AppMutex"       = $appMutex
        "AppExeName"     = $appExeName
        "ResourcesDir"   = "$innoDir"
        "ShellNameShort" = $appShellNameShort
        "AppUserId"      = $appUserId
        "UrlScheme"      = $env:ZED_PRODUCT_URL_SCHEME
        "Version"        = "$env:RELEASE_VERSION"
        "NumericVersion" = ($env:RELEASE_VERSION -split '[-+]')[0]
        "SourceDir"      = "$env:ZED_WORKSPACE"
    }
    if ($Sim) {
        $definitions["Sim"] = "1"
    }

    if (($env:ZED_PRODUCT_APP_FEATURES -split ',') -contains 'agentic-tools') {
        $definitions["ManagedRuntime"] = "1"
    }
    $defs = @()
    foreach ($key in $definitions.Keys) {
        $defs += "/d$key=`"$($definitions[$key])`""
    }

    $innoArgs = @($issFilePath) + $defs
    if (Test-Path Env:ZED_SIGN_BUNDLE) { Remove-Item Env:ZED_SIGN_BUNDLE -ErrorAction Stop }
    if($canCodeSign) {
        # Checked by zed.iss to decide whether to sign the installer.
        $env:ZED_SIGN_BUNDLE = "1"
        $signTool = "pwsh.exe -NoProfile -ExecutionPolicy Bypass -File `$q$innoDir\sign.ps1`$q `$f"
        $innoArgs += "/sDefaultsign=`"$signTool`""
    }

    # Execute Inno Setup
    Write-Host "🚀 Running Inno Setup: $innoSetupPath $innoArgs"
    $process = Start-Process -FilePath $innoSetupPath -ArgumentList $innoArgs -NoNewWindow -Wait -PassThru

    if ($process.ExitCode -eq 0) {
        $expectedSetupPath = "$CargoTargetDir\release\$appSetupName.exe"
        if (-not (Test-Path $expectedSetupPath -PathType Leaf)) {
            throw "Inno Setup did not produce the expected product artifact: $expectedSetupPath"
        }
        if ($canCodeSign) {
            & "$innoDir\sign.ps1" -filePath $expectedSetupPath -VerifyOnly
        }
        Write-Host "✅ Inno Setup successfully compiled the installer"
        Write-Output "SETUP_PATH=$CargoTargetDir/release/$appSetupName.exe" >> $env:GITHUB_ENV
        $script:buildSuccess = $true
    }
    else {
        Write-Host "❌ Inno Setup failed: $($process.ExitCode)"
        $script:buildSuccess = $false
    }
}

ParseZedWorkspace
$innoDir = "$env:ZED_WORKSPACE\inno\$Architecture"
$debugArchive = "$CargoOutDir\$env:ZED_PRODUCT_ID-$env:RELEASE_VERSION-$env:ZED_RELEASE_CHANNEL.dbg.zip"

CheckEnvironmentVariables
PrepareForBundle
if ($bundlePhase -eq 'all' -or $bundlePhase -eq 'application') {
    GenerateLicenses
    BuildProductBinaries
}
if ($bundlePhase -eq 'all' -or $bundlePhase -eq 'remote-server') {
    BuildRemoteServer
}
if ($bundlePhase -eq 'application') {
    foreach ($binary in @('zed.exe', 'cli.exe', 'auto_update_helper.exe')) {
        if (-not (Test-Path "$CargoOutDir\$binary" -PathType Leaf)) {
            throw "Application build did not produce $binary"
        }
    }
    exit 0
}
if ($bundlePhase -eq 'remote-server') {
    if (-not (Test-Path "$CargoOutDir\remote_server.exe" -PathType Leaf)) {
        throw "Remote-server build did not produce remote_server.exe"
    }
    exit 0
}
if (-not (Test-Path "assets/licenses.md" -PathType Leaf)) {
    throw "The generated license file is missing"
}
foreach ($binary in @('zed.exe', 'cli.exe', 'auto_update_helper.exe', 'remote_server.exe')) {
    if (-not (Test-Path "$CargoOutDir\$binary" -PathType Leaf)) {
        throw "Packaging input is missing $binary"
    }
}
PackageRemoteServer
StageProductBinaries
ZipProductDebugSymbols
DownloadAMDGpuServices
DownloadConpty
CollectFiles
if (($env:ZED_PRODUCT_APP_FEATURES -split ',') -contains 'agentic-tools') {
    python script/bundle-managed-runtime --platform "windows-$Architecture" --destination "$innoDir/llama-cpp"
    if ($LASTEXITCODE -ne 0) { throw "Managed runtime packaging failed" }
}
SignZedAndItsFriends
BuildInstaller

if ($buildSuccess) {
    Write-Output "Build successful"
    if ($Install) {
        Write-Output "Installing $env:ZED_PRODUCT_DISPLAY_NAME..."
        Start-Process -FilePath "$CargoTargetDir/release/$env:ZED_PRODUCT_ARTIFACT_NAME"
    }
    exit 0
}
else {
    Write-Output "Build failed"
    exit 1
}
