{ lib, stdenvNoCC }:

stdenvNoCC.mkDerivation {
  pname = "zed-sim-backend-mlu-metadata";
  version = "neuware1.20-abi1";

  dontUnpack = true;

  installPhase = ''
    runHook preInstall
    destination="$out/lib/zed/sim-backends/mlu"
    mkdir -p "$destination/abi"
    cp ${../../../crates/sim_backend_mlu/abi/symbols-v1.json} "$destination/abi/"
    cp ${../../../crates/sim_backend_mlu/LICENSES} "$destination/"
    cp ${./package-policy.json} "$destination/"
    runHook postInstall
  '';

  passthru = {
    redistributesVendorRuntime = false;
    invokesVendorCompiler = false;
    releasePackager = "script/package-sim-backend-mlu";
    cryptographicVerificationOwner = "sim_runtime::NativeFfiRegistry";
    supportedTargets = [ "x86_64-linux" "aarch64-linux" ];
  };

  meta = {
    description = "Reviewed Zed Cambricon MLU ABI metadata without vendor runtime redistribution";
    license = lib.licenses.gpl3Plus;
    platforms = [ "x86_64-linux" "aarch64-linux" ];
  };
}
