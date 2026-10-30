param (
    [Parameter(Mandatory = $true)]
    [string]$filePath,
    [switch]$VerifyOnly
)

Set-PSDebug -Off
$ErrorActionPreference = 'Stop'

function Verify-SignedFile([string]$path) {
    $signature = Get-AuthenticodeSignature -LiteralPath $path
    if ($signature.Status -ne 'Valid' -or $null -eq $signature.SignerCertificate -or $null -eq $signature.TimeStamperCertificate) {
        throw "A valid Authenticode signature and trusted timestamp are required: $path"
    }
    if ([string]::IsNullOrWhiteSpace($env:WINDOWS_SIGNING_PUBLISHER) -or $signature.SignerCertificate.Subject -cne $env:WINDOWS_SIGNING_PUBLISHER) {
        throw "The signature does not match WINDOWS_SIGNING_PUBLISHER: $path"
    }
    $codeSigningOid = '1.3.6.1.5.5.7.3.3'
    if ($signature.SignerCertificate.EnhancedKeyUsageList.ObjectId.Value -notcontains $codeSigningOid) {
        throw "The signer certificate is not valid for code signing: $path"
    }
    Write-Host "Verified signed payload: $path; approved publisher matched; certificate thumbprint=$($signature.SignerCertificate.Thumbprint); timestamp verified"
}

if ($VerifyOnly) {
    if (Test-Path -LiteralPath $filePath -PathType Container) {
        $files = @(Get-ChildItem -LiteralPath $filePath -Recurse -File | Where-Object { $_.Extension -in '.exe', '.dll' })
        if ($files.Count -eq 0) { throw "No installed native payloads were found" }
        foreach ($file in $files) { Verify-SignedFile $file.FullName }
    } else {
        Verify-SignedFile $filePath
    }
    exit 0
}

Import-Module TrustedSigning -RequiredVersion 0.5.8 -ErrorAction Stop
$params = @{}

$endpoint = $ENV:ENDPOINT
if ([string]::IsNullOrWhiteSpace($endpoint)) {
    throw "The 'ENDPOINT' env is required."
}
$params["Endpoint"] = $endpoint

$trustedSigningAccountName = $ENV:ACCOUNT_NAME
if ([string]::IsNullOrWhiteSpace($trustedSigningAccountName)) {
    throw "The 'ACCOUNT_NAME' env is required."
}
$params["CodeSigningAccountName"] = $trustedSigningAccountName

$certificateProfileName = $ENV:CERT_PROFILE_NAME
if ([string]::IsNullOrWhiteSpace($certificateProfileName)) {
    throw "The 'CERT_PROFILE_NAME' env is required."
}
$params["CertificateProfileName"] = $certificateProfileName

$fileDigest = $ENV:FILE_DIGEST
if ([string]::IsNullOrWhiteSpace($fileDigest)) {
    throw "The 'FILE_DIGEST' env is required."
}
$params["FileDigest"] = $fileDigest

$timeStampDigest = $ENV:TIMESTAMP_DIGEST
if ([string]::IsNullOrWhiteSpace($timeStampDigest)) {
    throw "The 'TIMESTAMP_DIGEST' env is required."
}
$params["TimestampDigest"] = $timeStampDigest

$timeStampServer = $ENV:TIMESTAMP_SERVER
if ([string]::IsNullOrWhiteSpace($timeStampServer)) {
    throw "The 'TIMESTAMP_SERVER' env is required."
}
$params["TimestampRfc3161"] = $timeStampServer

$params["Files"] = $filePath

if ($fileDigest -ne 'SHA256' -or $timeStampDigest -ne 'SHA256') {
    throw 'Production signing requires SHA256 file and timestamp digests'
}
if ($timeStampServer -ne 'http://timestamp.acs.microsoft.com') {
    throw 'Production signing requires the Azure RFC3161 timestamp service'
}
# Do not allow the SDK to borrow a runner or developer account's cached login.
foreach ($credential in @('WorkloadIdentity', 'ManagedIdentity', 'SharedTokenCache', 'VisualStudio', 'VisualStudioCode', 'AzureCli', 'AzurePowerShell', 'AzureDeveloperCli', 'InteractiveBrowser')) {
    $params["Exclude${credential}Credential"] = $true
}
Invoke-TrustedSigning @params -ErrorAction Stop
foreach ($path in $filePath.Split(',')) {
    Verify-SignedFile $path
}
