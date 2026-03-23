"""
Seed FalkorDB with all existing system data.

Populates: model versions, dataset images, demo patient,
existing scans, sample doctors, and audit logs.
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Ensure app is importable
sys.path.insert(0, os.path.dirname(__file__))

from app.services.graph_db import get_falkordb
from app.config import get_settings

settings = get_settings()
gdb = get_falkordb()


def seed_schema():
    """Re-initialize schema (idempotent — uses MERGE/CREATE INDEX IF NOT EXISTS)."""
    print("[1/7] Initializing schema...")
    gdb.initialize_schema()
    print("      Schema ready (indexes + tumor types/grades seeded).")


def seed_model_versions():
    """Register the 4 pre-trained model files as ModelVersion nodes."""
    print("[2/7] Registering model versions...")

    models = [
        ("BrainTumorCNN", "1.0.0", 0.92, settings.model_2d_path),
        ("ResNet50", "1.0.0", 0.95, settings.model_resnet50_path),
        ("EfficientNet-B0", "1.0.0", 0.94, settings.model_efficientnet_path),
        ("DenseNet-121", "1.0.0", 0.93, settings.model_densenet_path),
    ]

    count = 0
    for model_name, version, accuracy, path in models:
        existing = gdb.get_model_versions(model_name)
        if existing:
            print(f"      {model_name} v{version} already exists, skipping.")
            continue
        gdb.create_model_version(model_name, version, accuracy, path)
        count += 1
        print(f"      + {model_name} v{version} (acc={accuracy:.0%}, path={path})")

    print(f"      {count} model version(s) registered.")


def seed_dataset_metadata():
    """Scan data/combined/ and register every image as a DatasetImage node."""
    print("[3/7] Registering dataset images...")

    # Resolve combined dir relative to backend/
    combined_dir = Path(settings.combined_dataset_dir)
    if not combined_dir.is_absolute():
        combined_dir = Path(__file__).parent / combined_dir

    if not combined_dir.exists():
        print(f"      Dataset directory not found: {combined_dir}")
        print("      Skipping dataset metadata (run downloader first).")
        return

    images = []
    for split in ["train", "val"]:
        split_dir = combined_dir / split
        if not split_dir.exists():
            continue
        for cls_dir in split_dir.iterdir():
            if cls_dir.is_dir():
                for img_path in cls_dir.iterdir():
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        images.append({
                            "path": str(img_path),
                            "class_label": cls_dir.name,
                            "split": split,
                        })

    if not images:
        print("      No images found in dataset directories.")
        print("      (Download dataset first to populate data/combined/)")
        return

    gdb.store_dataset_metadata(images, "combined")
    # Count per class
    class_counts = {}
    for img in images:
        key = f"{img['split']}/{img['class_label']}"
        class_counts[key] = class_counts.get(key, 0) + 1
    for k, v in sorted(class_counts.items()):
        print(f"      {k}: {v}")
    print(f"      {len(images)} dataset image(s) registered.")


def seed_demo_patient_and_scans():
    """Register the demo patient and all existing uploaded scans."""
    print("[4/7] Registering demo patient and existing scans...")

    patient = gdb.get_or_create_demo_patient()
    patient_mrn = patient["mrn"]
    print(f"      Patient: {patient_mrn}")

    # Find existing scan files — check all patient folders under scans/
    uploads_dir = Path(settings.local_storage_dir)
    if not uploads_dir.is_absolute():
        uploads_dir = Path(__file__).parent / uploads_dir

    scans_root = uploads_dir / "scans"
    if not scans_root.exists():
        print("      No scans directory found.")
        return

    # Discover all patient scan folders (e.g. DEMO-0001, DEMO-001)
    patient_dirs = [d for d in scans_root.iterdir() if d.is_dir()]
    if not patient_dirs:
        print("      No existing scans found.")
        return

    # Use the first patient folder found (map files to the DB patient MRN)
    scans_dir = patient_dirs[0]
    folder_mrn = scans_dir.name
    print(f"      Found scan folder: {folder_mrn} (mapped to patient {patient_mrn})")

    count = 0
    for scan_file in scans_dir.iterdir():
        if not scan_file.is_file():
            continue
        # Extract UUID from filename: scan_{uuid}_{original_name}
        name = scan_file.name
        if not name.startswith("scan_"):
            continue
        parts = name.split("_", 2)  # ["scan", uuid, rest]
        if len(parts) < 3:
            continue
        scan_id = parts[1]

        # Verify it's a valid UUID
        try:
            uuid.UUID(scan_id)
        except ValueError:
            continue

        storage_path = f"scans/{folder_mrn}/{name}"

        # Check if scan already exists
        existing = gdb.get_scan(scan_id)
        if existing:
            continue

        gdb.create_scan(scan_id, patient_mrn, ["image"], storage_location=storage_path)
        gdb.tag_scan(scan_id, "brain_mri")
        gdb.tag_scan(scan_id, "uploaded")
        count += 1
        print(f"      + Scan {scan_id[:8]}... ({parts[2][:40]})")

    print(f"      {count} scan(s) registered.")


def seed_doctors():
    """Create sample radiologist records."""
    print("[5/7] Registering doctors...")

    doctors = [
        ("DR-001", "Dr. Sarah Chen", "Neuroradiology", "NR-2024-001", "s.chen@hospital.org"),
        ("DR-002", "Dr. Ahmed Hassan", "Diagnostic Radiology", "DR-2024-002", "a.hassan@hospital.org"),
        ("DR-003", "Dr. Maria Garcia", "Neuropathology", "NP-2024-003", "m.garcia@hospital.org"),
    ]

    count = 0
    for doc_id, name, spec, lic, email in doctors:
        existing = gdb.get_doctor(doc_id)
        if existing:
            print(f"      {name} already exists, skipping.")
            continue
        gdb.create_doctor(doc_id, name, spec, lic, email)
        count += 1
        print(f"      + {name} ({spec})")

    # Assign Dr. Chen to DEMO patient
    patient = gdb.get_or_create_demo_patient()
    gdb.assign_doctor_to_patient("DR-001", patient["mrn"], notes="Primary reviewing radiologist")
    print(f"      {count} doctor(s) registered. Dr. Chen assigned to {patient['mrn']}.")


def seed_audit_logs():
    """Create initial audit trail entries."""
    print("[6/7] Creating audit log entries...")

    logs = [
        ("SYSTEM_INIT", "System", "system", "system", "FalkorDB schema initialized and seeded"),
        ("MODEL_REGISTERED", "ModelVersion", "all", "system", "4 pre-trained models registered"),
        ("PATIENT_CREATED", "Patient", "DEMO-0001", "system", "Demo patient created for development"),
        ("DOCTOR_REGISTERED", "Doctor", "DR-001,DR-002,DR-003", "system", "3 radiologists registered"),
    ]

    for action, etype, eid, actor, details in logs:
        gdb.create_audit_log(action, etype, eid, actor, details)

    print(f"      {len(logs)} audit log entries created.")


def verify():
    """Query and display final database contents."""
    print("[7/7] Verifying database contents...")
    print()

    from falkordb import FalkorDB
    db = FalkorDB(host=settings.falkordb_host, port=settings.falkordb_port)
    g = db.select_graph("brain_tumor")

    r = g.query("MATCH (n) RETURN labels(n) AS label, count(n) AS cnt ORDER BY cnt DESC")
    total_nodes = 0
    print("  ┌─────────────────────────────┬───────┐")
    print("  │ Node Label                  │ Count │")
    print("  ├─────────────────────────────┼───────┤")
    for row in r.result_set:
        label = str(row[0]).strip("[]'")
        cnt = row[1]
        total_nodes += cnt
        print(f"  │ {label:<27} │ {cnt:>5} │")
    print("  ├─────────────────────────────┼───────┤")
    print(f"  │ {'TOTAL NODES':<27} │ {total_nodes:>5} │")
    print("  └─────────────────────────────┴───────┘")

    r2 = g.query("MATCH ()-[r]->() RETURN type(r) AS rel, count(r) AS cnt ORDER BY cnt DESC")
    total_rels = 0
    print()
    print("  ┌─────────────────────────────┬───────┐")
    print("  │ Relationship Type           │ Count │")
    print("  ├─────────────────────────────┼───────┤")
    for row in r2.result_set:
        total_rels += row[1]
        print(f"  │ {row[0]:<27} │ {row[1]:>5} │")
    print("  ├─────────────────────────────┼───────┤")
    print(f"  │ {'TOTAL RELATIONSHIPS':<27} │ {total_rels:>5} │")
    print("  └─────────────────────────────┴───────┘")


if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  FalkorDB Seeder — Brain Tumor AI Framework")
    print("=" * 55)
    print()

    seed_schema()
    seed_model_versions()
    seed_dataset_metadata()
    seed_demo_patient_and_scans()
    seed_doctors()
    seed_audit_logs()
    verify()

    print()
    print("Seeding complete!")
