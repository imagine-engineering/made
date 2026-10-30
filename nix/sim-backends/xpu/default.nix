{ lib, stdenvNoCC }:

stdenvNoCC.mkDerivation {
  pname = "zed-sim-backend-xpu-metadata";
  version = "level-zero-1.11.0-onednn-3.5.0-abi1";

  dontUnpack = true;

  installPhase = ''
    runHook preInstall
    destination="$out/lib/zed/sim-backends/xpu"
    mkdir -p "$destination/abi"
    cp ${../../../crates/sim_backend_xpu/abi/symbols-v1.json} "$destination/abi/"
    cp ${../../../crates/sim_backend_xpu/abi/reviewed-execution-bindings-v1.txt} "$destination/abi/"
    cp ${../../../crates/sim_backend_xpu/abi/verify-execution-bindings.c} "$destination/abi/"
    cp ${../../../crates/sim_backend_xpu/LICENSES} "$destination/"
    cp ${./package-policy.json} "$destination/"
    runHook postInstall
  '';

  passthru = {
    redistributesVendorRuntime = false;
    invokesVendorCompiler = false;
    releasePackager = "script/package-sim-backend-xpu";
    cryptographicVerificationOwner = "sim_runtime::NativeFfiRegistry";
    supportedTargets = [ "x86_64-linux" "x86_64-windows" ];
  };

  meta = {
    description = "Reviewed Zed Intel XPU ABI metadata without vendor runtime redistribution";
    license = lib.licenses.gpl3Plus;
    platforms = [ "x86_64-linux" "x86_64-windows" ];
  };
}
