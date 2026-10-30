{ lib, stdenvNoCC }:

stdenvNoCC.mkDerivation {
  pname = "zed-sim-backend-rocm-metadata";
  version = "6.1.0-abi1";

  dontUnpack = true;

  installPhase = ''
    runHook preInstall
    destination="$out/lib/zed/sim-backends/rocm"
    mkdir -p "$destination/abi"
    cp ${../../../crates/sim_backend_rocm/abi/symbols-v1.json} "$destination/abi/"
    cp ${../../../crates/sim_backend_rocm/LICENSES} "$destination/"
    cp ${./package-policy.json} "$destination/"
    runHook postInstall
  '';

  passthru = {
    redistributesAmdRuntime = false;
    requiresPlatformPackageSignature = true;
  };

  meta = {
    description = "Reviewed Zed ROCm ABI metadata without AMD runtime redistribution";
    license = lib.licenses.gpl3Plus;
    platforms = [ "x86_64-linux" ];
  };
}
