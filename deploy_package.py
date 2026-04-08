import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "deploy_bundle"

ARTIFACT_PATHS = [
    "sampled_data/graph_data_plain.pt",
    "sampled_data/graphsage_model.pth",
    "sampled_data/tfidf_vectorizer.joblib",
    "sampled_data/label_encoder_mapping.joblib",
    "sampled_data/model_metadata.pkl",
    "sampled_data/pipeline_metadata.pkl",
]

CODE_PATHS = [
    "app/main.py",
    "app/static/index.html",
    "src/predict.py",
    "src/model.py",
    "requirements.txt",
    "README.md",
]


def copy_paths(paths):
    for rel_path in paths:
        src = ROOT / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Required file missing: {rel_path}")
        dst = DEST / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_manifest():
    manifest_path = DEST / "MANIFEST.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("Deployment bundle manifest\n")
        f.write("==========================\n\n")
        f.write("Included artifact files:\n")
        for path in ARTIFACT_PATHS:
            f.write(f"- {path}\n")
        f.write("\nIncluded application files:\n")
        for path in CODE_PATHS:
            f.write(f"- {path}\n")


def main():
    if DEST.exists():
        print(f"Removing existing deployment bundle at: {DEST}")
        shutil.rmtree(DEST)
    print(f"Creating deployment bundle in: {DEST}")
    copy_paths(ARTIFACT_PATHS)
    copy_paths(CODE_PATHS)
    write_manifest()
    print("Deployment bundle created successfully.")
    print(f"You can now package or deploy the folder: {DEST}")


if __name__ == "__main__":
    main()
