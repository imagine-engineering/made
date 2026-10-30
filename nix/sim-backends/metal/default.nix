{ lib, stdenvNoCC }:

stdenvNoCC.mkDerivation {
  pname = "zed-sim-backend-metal-metadata";
  version = "macos13-metal3-abi1";

  dontUnpack = true;

  installPhase = ''
    runHook preInstall
    destination="$out/lib/zed/sim-backends/metal"
    mkdir -p "$destination/abi" "$destination/kernels"
    cp ${../../../crates/sim_backend_metal/abi/symbols-v1.json} "$destination/abi/"
    cp ${../../../crates/sim_backend_metal/abi/execution-v1.json} "$destination/abi/"
    cp ${../../../crates/sim_backend_metal/abi/reviewed-execution-bindings-v1.txt} "$destination/abi/"
    cp ${../../../crates/sim_backend_metal/kernels/readiness.metal} "$destination/kernels/"
    cp ${../../../crates/sim_backend_metal/kernels/tensor_ops.metal} "$destination/kernels/"
    cp ${../../../crates/sim_backend_metal/LICENSES} "$destination/"
    cp ${../../../crates/sim_backend_metal/LICENSES.execution} "$destination/"
    cp ${./package-policy.json} "$destination/"
    cp ${./execution-policy.json} "$destination/"
    cp ${./ffi-contracts-v1.schema.json} "$destination/"
    runHook postInstall
  '';

  passthru = {
    redistributesAppleFrameworks = false;
    invokesXcodeDuringNixBuild = false;
    releasePackager = "script/package-sim-backend-metal";
    cryptographicVerificationOwner = "sim_runtime::MetalPackageVerificationKey";
    structuralMetadataAuthorizesExecution = false;
  };

  meta = {
    description = "Reviewed Zed Metal ABI and kernel-source metadata without Apple framework redistribution";
    license = lib.licenses.gpl3Plus;
    platforms = [ "aarch64-darwin" "x86_64-darwin" ];
  };
}
