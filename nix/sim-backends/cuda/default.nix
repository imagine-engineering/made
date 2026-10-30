{ lib, stdenvNoCC }:

stdenvNoCC.mkDerivation {
  pname = "zed-sim-backend-cuda-metadata";
  version = "cuda12.2-cudnn9-abi1";

  dontUnpack = true;

  installPhase = ''
    runHook preInstall
    destination="$out/lib/zed/sim-backends/cuda"
    mkdir -p "$destination/abi" "$destination/kernels"
    cp ${../../../crates/sim_backend_cuda/abi/symbols-v1.json} "$destination/abi/"
    cp ${../../../crates/sim_backend_cuda/kernels/core-v1.ptx} "$destination/kernels/"
    cp ${../../../crates/sim_backend_cuda/kernels/manifest-v1.json} "$destination/kernels/"
    cp ${../../../crates/sim_backend_cuda/LICENSES} "$destination/"
    cp ${./package-policy.json} "$destination/"
    runHook postInstall
  '';

  passthru = {
    redistributesDriver = false;
    redistributesVendorRuntime = false;
    invokesVendorCompiler = false;
    releasePackager = "script/package-sim-backend-cuda";
    cryptographicVerificationOwner = "sim_runtime::NativeFfiRegistry";
    supportedTargets = [ "x86_64-linux" "aarch64-linux" "x86_64-windows" ];
  };

  meta = {
    description = "Reviewed Zed NVIDIA CUDA ABI and PTX metadata without vendor runtime redistribution";
    license = lib.licenses.gpl3Plus;
    platforms = [ "x86_64-linux" "aarch64-linux" "x86_64-windows" ];
  };
}
