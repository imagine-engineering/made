
function ParseZedWorkspace {
    $metadata = cargo metadata --no-deps --offline | ConvertFrom-Json
    $env:ZED_WORKSPACE = $metadata.workspace_root
    if ([string]::IsNullOrWhiteSpace($env:RELEASE_VERSION)) {
        $env:RELEASE_VERSION = $metadata.packages | Where-Object { $_.name -eq "zed" } | Select-Object -ExpandProperty version
    }
}
