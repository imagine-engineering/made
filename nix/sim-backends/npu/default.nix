{ lib, stdenvNoCC }:

stdenvNoCC.mkDerivation {
  pname = "zed-sim-backend-npu-metadata";
  version = "cann-8.0-rc3-abi1";

  dontUnpack = true;

  installPhase = ''
    runHook preInstall
    destination="$out/lib/zed/sim-backends/npu"
    mkdir -p "$destination/abi"
    cp ${../../../crates/sim_backend_npu/abi/symbols-v1.json} "$destination/abi/"
    cp ${../../../crates/sim_backend_npu/LICENSES} "$destination/"
    cp ${./package-policy.json} "$destination/"
    runHook postInstall
  '';

  passthru = {
    redistributesCann = false;
    requiredExternalLibraries = [ "libascendcl.so" "libruntime.so" ];
    discoveryOrder = [ "SIM_ASCEND_ROOT" "ASCEND_HOME_PATH" "signed_package_roots" ];
    releasePackager = "script/package-sim-backend-npu";
    cryptographicVerificationOwner = "sim_runtime::NativeFfiRegistry plus signer-bound NPU package key";
  };

  meta = {
    description = "Reviewed Zed Huawei Ascend NPU ABI metadata without CANN redistribution";
    license = lib.licenses.gpl3Plus;
    platforms = [ "aarch64-linux" "x86_64-linux" ];
  };
}
