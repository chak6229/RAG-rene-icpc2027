"""Formal, label-free registry for the VAC-MQR risk module.

The registry owns the risk component only.  It does not select candidates,
consume gold labels, issue a risk certificate, or change the verification
policy pointer.  Every version is bound to its immutable fold artifacts and
to the protected upstream CURRENT files so a rollback cannot silently mix
components from different protocols.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit, logit, logsumexp

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "vac_mqr_risk_module"
FORMAT = "vac-mqr-risk-module-v1"
ALIASES = {
    "current": "current",
    "anchor-stage": "anchor-stage-threeway-20261016",
    "anchor_stage": "anchor-stage-threeway-20261016",
    "anchor-stage-threeway": "anchor-stage-threeway-20261016",
    "threeway-anchor": "anchor-stage-threeway-20261016",
    "threeway_anchor": "anchor-stage-threeway-20261016",
    "previous": "selected-structural-20261016",
    "structural": "selected-structural-20261016",
}


def sha(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _root_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _version_path(version: str) -> Path:
    if Path(version).name != version or version in ("", ".", ".."):
        raise ValueError("version must be a registered name")
    return CONFIG / "versions" / f"{version}.json"


def _check_hashes(entries, label: str) -> None:
    if not isinstance(entries, dict):
        raise ValueError(f"{label} must be a mapping")
    for raw_path, expected in entries.items():
        path = _root_path(raw_path)
        if not path.is_file():
            raise ValueError(f"{label} file is missing: {raw_path}")
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"{label} hash mismatch: {raw_path}")


def _check_artifacts(manifest: dict) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("manifest has no model artifacts")
    for key, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid artifact record: {key}")
        for field in ("path", "sha256", "family", "fold"):
            if field not in record:
                raise ValueError(f"artifact {key} missing {field}")
        path = _root_path(record["path"])
        if sha(path) != record["sha256"]:
            raise ValueError(f"artifact hash mismatch: {key}")


def _validate_manifest(manifest: dict, manifest_path: Path) -> dict:
    if manifest.get("format") != FORMAT:
        raise ValueError("unsupported risk-module manifest format")
    version = manifest.get("version")
    if version != manifest_path.stem:
        raise ValueError("manifest version/path mismatch")
    if manifest.get("module") not in ("selected_structural", "threeway_anchor"):
        raise ValueError("unsupported risk-module kind")
    if manifest.get("risk_certified") is not False:
        raise ValueError("risk module cannot be marked risk certified by registry")
    if manifest.get("certificate_issued") is not False:
        raise ValueError("historical risk module cannot issue a certificate")
    _check_hashes(manifest.get("source_hashes", {}), "source")
    _check_hashes(manifest.get("protected_hashes", {}), "protected")
    _check_artifacts(manifest)
    dependency = manifest.get("dependencies", {})
    if not isinstance(dependency, dict):
        raise ValueError("dependencies must be a mapping")
    for name, record in dependency.items():
        if not isinstance(record, dict) or "path" not in record or "sha256" not in record:
            raise ValueError(f"invalid dependency record: {name}")
        path = _root_path(record["path"])
        if sha(path) != record["sha256"]:
            raise ValueError(f"dependency hash mismatch: {name}")
    previous = manifest.get("previous_version")
    if previous is not None:
        previous_manifest = manifest.get("previous_manifest")
        previous_sha = manifest.get("previous_manifest_sha256")
        if not previous_manifest or not previous_sha:
            raise ValueError("rollback target is not fully bound")
        path = _root_path(previous_manifest)
        if path.stem != previous or sha(path) != previous_sha:
            raise ValueError("rollback manifest hash mismatch")
    return manifest


def resolve(version: str = "current") -> dict:
    """Resolve a version and verify all files it references."""
    requested = ALIASES.get(version, version)
    if requested == "current":
        pointer_path = CONFIG / "CURRENT.json"
        pointer = read_json(pointer_path)
        if pointer.get("format") != "vac-mqr-risk-module-current-v1":
            raise ValueError("unsupported CURRENT format")
        path = _root_path(pointer["manifest"])
        if sha(path) != pointer.get("manifest_sha256"):
            raise ValueError("CURRENT manifest hash mismatch")
        if path.stem != pointer.get("version"):
            raise ValueError("CURRENT version mismatch")
    else:
        path = _version_path(requested)
    return _validate_manifest(read_json(path), path)


def manifest_path(manifest: dict) -> Path:
    return CONFIG / "versions" / f"{manifest['version']}.json"


def verify_rollback(version: str = "current") -> dict:
    """Verify the active version and its exact previous-version rollback target."""
    current = resolve(version)
    previous = current.get("previous_version")
    if not previous:
        raise ValueError(f"{current['version']} has no rollback target")
    previous_manifest = resolve(previous)
    expected_path = current["previous_manifest"]
    expected_sha = current["previous_manifest_sha256"]
    actual_path = manifest_path(previous_manifest)
    if expected_path.replace("\\", "/") != actual_path.relative_to(ROOT).as_posix():
        raise ValueError("rollback path does not resolve to previous version")
    if sha(actual_path) != expected_sha:
        raise ValueError("rollback target changed")
    return {
        "passed": True,
        "current_version": current["version"],
        "rollback_version": previous_manifest["version"],
        "current_manifest_sha256": sha(manifest_path(current)),
        "rollback_manifest_sha256": sha(actual_path),
        "protected_hashes": current["protected_hashes"],
    }


def verify_registry() -> dict:
    current = resolve("current")
    rollback = verify_rollback("current")
    versions = [current["version"], current["previous_version"]]
    return {
        "passed": True,
        "format": FORMAT,
        "current": current["version"],
        "versions": versions,
        "risk_certified": False,
        "deployment_policy_promoted": False,
        "rollback": rollback,
    }


def _feature_matrix(data, expected_features: int, expected_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(data, dict) or "x" not in data or "native" not in data:
        raise ValueError("prediction input requires x and native arrays")
    x = np.asarray(data["x"], dtype=float)
    native = np.asarray(data["native"], dtype=float)
    if x.ndim != 2 or x.shape[1] != expected_features:
        raise ValueError(f"x must have shape (n, {expected_features})")
    supplied_names = data.get("feature_names")
    if supplied_names is not None:
        if list(supplied_names) != expected_names:
            raise ValueError("feature_names do not match the registered schema")
    if native.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(native).all():
        raise ValueError("prediction inputs must be finite and aligned")
    return x, native


class RiskModule:
    """Load one frozen fold and expose only label-free probability scoring.

    This adapter scores an already assembled observable feature matrix.  It
    deliberately does not choose candidates, re-run retrieval/CE, or expose
    historical output traces as live predictions.
    """

    def __init__(self, version: str = "current", family: str = "fusion", fold: int = 0):
        if family not in ("rewrite", "fusion"):
            raise ValueError("family must be rewrite or fusion")
        if not isinstance(fold, int) or fold not in range(4):
            raise ValueError("fold must be 0..3")
        self.manifest = resolve(version)
        key = f"{family}/fold{fold}"
        record = self.manifest["artifacts"].get(key)
        if record is None:
            raise ValueError(f"no artifact registered for {key}")
        if record["family"] != family or record["fold"] != fold:
            raise ValueError("artifact identity mismatch")
        with _root_path(record["path"]).open("rb") as handle:
            bundle = pickle.load(handle)
        if bundle.get("family") != family or int(bundle.get("fold")) != fold:
            raise ValueError("serialized fold identity mismatch")
        if bundle.get("protocol_sha256") != self.manifest["protocol_sha256"]:
            raise ValueError("serialized bundle is bound to another protocol")
        models = bundle.get("models", {})
        if self.manifest["module"] not in models:
            raise ValueError("serialized bundle lacks registered model arm")
        self.version = self.manifest["version"]
        self.family = family
        self.fold = fold
        self.module = self.manifest["module"]
        self.model = models[self.module]
        self.feature_count = int(self.manifest["feature_count"])
        self.feature_names = list(self.manifest.get("feature_names", []))
        if len(self.feature_names) != self.feature_count:
            raise ValueError("manifest feature schema is incomplete")
        if len(self.model.get("mask", [])) != self.feature_count:
            raise ValueError("model feature schema mismatch")

    def predict(self, data: dict) -> dict:
        x, native = _feature_matrix(data, self.feature_count, self.feature_names)
        model = self.model
        if model.get("fallback"):
            if model.get("kind") == "threeway":
                classes = np.tile(np.asarray(model["class_prior"], dtype=float), (len(x), 1))
                harm = classes[:, 0]
                gain = classes[:, 2] * model["positive_magnitude"] - harm * model["negative_magnitude"]
                return {"p_harm": harm, "p_classes": classes, "expected_gain": gain}
            return {"p_harm": np.full(len(x), model["binary_prior"], dtype=float)}
        design = np.c_[np.ones(len(x)), (x - model["mean"]) / model["scale"] * model["mask"]]
        if model.get("kind") == "threeway":
            q = float(model["positive_given_nonharm"])
            eta = np.log(np.c_[np.clip(native, 1e-6, 1 - 1e-6),
                               (1 - np.clip(native, 1e-6, 1 - 1e-6)) * (1 - q),
                               (1 - np.clip(native, 1e-6, 1 - 1e-6)) * q])
            beta = np.asarray(model["beta"], dtype=float)
            eta[:, [0, 2]] += design @ beta.T
            classes = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
            harm = classes[:, 0]
            gain = classes[:, 2] * model["positive_magnitude"] - harm * model["negative_magnitude"]
            return {"p_harm": harm, "p_classes": classes, "expected_gain": gain}
        beta = np.asarray(model["beta"], dtype=float)
        probability = expit(logit(np.clip(native, 1e-6, 1 - 1e-6)) + design @ beta)
        return {"p_harm": probability}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="current")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-rollback", action="store_true")
    args = parser.parse_args()
    if args.verify_rollback:
        print(json.dumps(verify_rollback(args.version), indent=2, sort_keys=True))
    elif args.verify:
        print(json.dumps(verify_registry(), indent=2, sort_keys=True))
    else:
        print(json.dumps(resolve(args.version), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
