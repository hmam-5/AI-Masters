"""
FalkorDB graph database service — sole persistence layer.

Stores patients, scans, inference jobs, classification/segmentation results,
training metadata, doctors, audit logs, and review assignments in a graph structure.

Graph Schema:
  (:Patient {mrn, sex, dob, created_at})
  (:Scan {id, date, modalities, storage_location, status})
  (:InferenceJob {id, scan_id, status, progress, started_at, completed_at, error_message, celery_task_id, created_at})
  (:AnalysisResult {job_id, confidence, grade, timestamp, details, models_loaded, agreement_score})
  (:TumorType {name, description})
  (:TumorGrade {grade, severity, description})
  (:SubregionResult {subregion, confidence, volume_mm3, mask_storage_path})
  (:ClassificationResult {job_id, tumor_grade, confidence_score, classification_details, created_at})
  (:DatasetImage {path, class_label, split, source_dataset})
  (:TrainingRun {id, accuracy, epochs, timestamp, model_path})
  (:Doctor {id, name, specialization, license_number, email, created_at})
  (:AuditLog {id, action, entity_type, entity_id, actor, timestamp, details})
  (:ModelVersion {id, model_name, version, accuracy, path, created_at, status})
  (:Tag {name})

Many-to-Many relationships:
  (Doctor)-[:REVIEWED {reviewed_at, notes, decision}]->(Patient)  — M:N
  (Doctor)-[:ASSIGNED_TO {assigned_at, priority, status}]->(InferenceJob) — M:N
  (Scan)-[:TAGGED_WITH]->(Tag) — M:N
  (ModelVersion)-[:SUPERSEDES]->(ModelVersion) — version chain
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

from falkordb import FalkorDB

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class FalkorDBService:
    """Service for interacting with FalkorDB graph database."""

    def __init__(self, host: Optional[str] = None, port: Optional[int] = None):
        self.host = host or settings.falkordb_host
        self.port = port or settings.falkordb_port
        self._db = None
        self._graph = None

    @property
    def db(self) -> FalkorDB:
        if self._db is None:
            self._db = FalkorDB(host=self.host, port=self.port)
        return self._db

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.db.select_graph("brain_tumor")
        return self._graph

    # ── Schema ────────────────────────────────────────────────────

    def initialize_schema(self) -> None:
        """Create indexes and seed tumor type/grade nodes + new entities."""
        try:
            self.graph.query("CREATE INDEX FOR (p:Patient) ON (p.mrn)")
            self.graph.query("CREATE INDEX FOR (s:Scan) ON (s.id)")
            self.graph.query("CREATE INDEX FOR (j:InferenceJob) ON (j.id)")
            self.graph.query("CREATE INDEX FOR (a:AnalysisResult) ON (a.job_id)")
            self.graph.query("CREATE INDEX FOR (c:ClassificationResult) ON (c.job_id)")
            self.graph.query("CREATE INDEX FOR (d:DatasetImage) ON (d.path)")
            self.graph.query("CREATE INDEX FOR (t:TrainingRun) ON (t.id)")
            self.graph.query("CREATE INDEX FOR (doc:Doctor) ON (doc.id)")
            self.graph.query("CREATE INDEX FOR (al:AuditLog) ON (al.id)")
            self.graph.query("CREATE INDEX FOR (mv:ModelVersion) ON (mv.id)")
            self.graph.query("CREATE INDEX FOR (tag:Tag) ON (tag.name)")
        except Exception:
            pass  # indexes may already exist

        tumor_types = [
            ("Glioma", "A tumor that occurs in the brain and spinal cord"),
            ("Meningioma", "A tumor that arises from the meninges"),
            ("Pituitary", "A tumor that forms in the pituitary gland"),
        ]
        for name, desc in tumor_types:
            self.graph.query(
                "MERGE (t:TumorType {name: $name}) SET t.description = $desc",
                {"name": name, "desc": desc},
            )

        grade_info = [
            ("No Tumor", 0, "No tumor detected"),
            ("Grade I", 1, "Slow-growing, least aggressive"),
            ("Grade II", 2, "Low-grade, slow-growing but may recur"),
            ("Grade III", 3, "Anaplastic, moderately aggressive"),
            ("Grade IV", 4, "Glioblastoma, most aggressive"),
        ]
        for grade, severity, desc in grade_info:
            self.graph.query(
                "MERGE (g:TumorGrade {grade: $grade}) "
                "SET g.severity = $severity, g.description = $desc",
                {"grade": grade, "severity": severity, "desc": desc},
            )

        logger.info("FalkorDB schema initialized")

    # ── Patient CRUD ──────────────────────────────────────────────

    def create_patient(self, mrn: str, date_of_birth: str, sex: str | None = None) -> dict:
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "MERGE (p:Patient {mrn: $mrn}) "
            "ON CREATE SET p.dob = $dob, p.sex = $sex, p.created_at = $now, p.updated_at = $now "
            "ON MATCH SET p.updated_at = $now",
            {"mrn": mrn, "dob": date_of_birth, "sex": sex or "Unknown", "now": now},
        )
        return self.get_patient(mrn)

    def get_patient(self, mrn: str) -> dict | None:
        result = self.graph.query(
            "MATCH (p:Patient {mrn: $mrn}) "
            "RETURN p.mrn, p.dob, p.sex, p.created_at, p.updated_at",
            {"mrn": mrn},
        )
        if not result.result_set:
            return None
        r = result.result_set[0]
        return {"mrn": r[0], "date_of_birth": r[1], "sex": r[2], "created_at": r[3], "updated_at": r[4]}

    def get_or_create_demo_patient(self) -> dict:
        p = self.get_patient("DEMO-001")
        if not p:
            p = self.create_patient("DEMO-001", "1990-01-01T00:00:00", "Unknown")
        return p

    # ── Scan CRUD ─────────────────────────────────────────────────

    def create_scan(self, scan_id: str, patient_mrn: str, modalities: list[str], storage_location: str = "") -> dict:
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "CREATE (s:Scan {id: $id, date: $date, modalities: $mods, "
            "  storage_location: $loc, status: 'uploaded'})",
            {"id": scan_id, "date": now, "mods": json.dumps(modalities), "loc": storage_location},
        )
        self.graph.query(
            "MATCH (p:Patient {mrn: $mrn}), (s:Scan {id: $sid}) "
            "MERGE (p)-[:HAS_SCAN]->(s)",
            {"mrn": patient_mrn, "sid": scan_id},
        )
        return {"id": scan_id, "patient_mrn": patient_mrn, "modalities": modalities, "storage_location": storage_location, "date": now}

    def get_scan(self, scan_id: str) -> dict | None:
        result = self.graph.query(
            "MATCH (s:Scan {id: $id}) "
            "OPTIONAL MATCH (p:Patient)-[:HAS_SCAN]->(s) "
            "RETURN s.id, s.date, s.modalities, s.storage_location, s.status, p.mrn",
            {"id": scan_id},
        )
        if not result.result_set:
            return None
        r = result.result_set[0]
        return {"id": r[0], "date": r[1], "modalities": r[2], "storage_location": r[3], "status": r[4], "patient_mrn": r[5]}

    def update_scan_storage(self, scan_id: str, storage_location: str) -> None:
        self.graph.query(
            "MATCH (s:Scan {id: $id}) SET s.storage_location = $loc",
            {"id": scan_id, "loc": storage_location},
        )

    # ── InferenceJob CRUD ─────────────────────────────────────────

    def create_job(self, job_id: str, scan_id: str) -> dict:
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "CREATE (j:InferenceJob {id: $id, scan_id: $scan_id, status: 'pending', "
            "  progress: 0, started_at: '', completed_at: '', error_message: '', "
            "  celery_task_id: '', created_at: $now, updated_at: $now})",
            {"id": job_id, "scan_id": scan_id, "now": now},
        )
        self.graph.query(
            "MATCH (s:Scan {id: $sid}), (j:InferenceJob {id: $jid}) "
            "MERGE (s)-[:HAS_JOB]->(j)",
            {"sid": scan_id, "jid": job_id},
        )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict | None:
        result = self.graph.query(
            "MATCH (j:InferenceJob {id: $id}) "
            "RETURN j.id, j.scan_id, j.status, j.progress, j.started_at, "
            "       j.completed_at, j.error_message, j.celery_task_id, j.created_at, j.updated_at",
            {"id": job_id},
        )
        if not result.result_set:
            return None
        r = result.result_set[0]
        return {
            "id": r[0], "scan_id": r[1], "status": r[2], "progress_percentage": r[3],
            "started_at": r[4] or None, "completed_at": r[5] or None,
            "error_message": r[6] or None, "celery_task_id": r[7] or None,
            "created_at": r[8], "updated_at": r[9],
        }

    def update_job(self, job_id: str, **fields) -> None:
        now = datetime.utcnow().isoformat()
        set_clauses = ["j.updated_at = $now"]
        params: dict = {"id": job_id, "now": now}
        field_map = {
            "status": "j.status",
            "progress": "j.progress",
            "progress_percentage": "j.progress",
            "started_at": "j.started_at",
            "completed_at": "j.completed_at",
            "error_message": "j.error_message",
            "celery_task_id": "j.celery_task_id",
        }
        for key, val in fields.items():
            if key in field_map:
                param_name = f"p_{key}"
                set_clauses.append(f"{field_map[key]} = ${param_name}")
                params[param_name] = str(val) if val is not None else ""
        cypher = f"MATCH (j:InferenceJob {{id: $id}}) SET {', '.join(set_clauses)}"
        self.graph.query(cypher, params)

    # ── Classification / Segmentation Results ─────────────────────

    def save_classification_result(self, job_id: str, tumor_grade: str, confidence_score: float, classification_details: dict | None = None) -> None:
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "CREATE (c:ClassificationResult {"
            "  job_id: $job_id, tumor_grade: $grade, confidence_score: $conf, "
            "  classification_details: $details, created_at: $now})",
            {
                "job_id": job_id,
                "grade": tumor_grade,
                "conf": confidence_score,
                "details": json.dumps(classification_details) if classification_details else "{}",
                "now": now,
            },
        )
        self.graph.query(
            "MATCH (j:InferenceJob {id: $jid}), (c:ClassificationResult {job_id: $jid}) "
            "MERGE (j)-[:HAS_CLASSIFICATION]->(c)",
            {"jid": job_id},
        )

    def get_classification_results(self, job_id: str) -> list[dict]:
        result = self.graph.query(
            "MATCH (c:ClassificationResult {job_id: $jid}) "
            "RETURN c.tumor_grade, c.confidence_score, c.classification_details, c.created_at",
            {"jid": job_id},
        )
        out = []
        for r in result.result_set:
            details = r[2]
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except Exception:
                    details = {}
            out.append({"tumor_grade": r[0], "confidence_score": r[1], "classification_details": details, "created_at": r[3]})
        return out

    def save_segmentation_result(self, job_id: str, subregion: str, confidence_score: float, volume_mm3: float, mask_storage_path: str = "") -> None:
        self.graph.query(
            "CREATE (sr:SegmentationResultNode {"
            "  job_id: $jid, subregion: $sub, confidence_score: $conf, "
            "  volume_mm3: $vol, mask_storage_path: $mask})",
            {"jid": job_id, "sub": subregion, "conf": confidence_score, "vol": volume_mm3, "mask": mask_storage_path},
        )
        self.graph.query(
            "MATCH (j:InferenceJob {id: $jid}), (sr:SegmentationResultNode {job_id: $jid, subregion: $sub}) "
            "MERGE (j)-[:HAS_SEGMENTATION_RESULT]->(sr)",
            {"jid": job_id, "sub": subregion},
        )

    def get_segmentation_results(self, job_id: str) -> list[dict]:
        result = self.graph.query(
            "MATCH (sr:SegmentationResultNode {job_id: $jid}) "
            "RETURN sr.subregion, sr.confidence_score, sr.volume_mm3, sr.mask_storage_path",
            {"jid": job_id},
        )
        return [
            {"subregion": r[0], "confidence_score": r[1], "volume_mm3": r[2], "mask_storage_path": r[3]}
            for r in result.result_set
        ]

    # ── Analysis Result (high-level, links to grades/types) ───────

    def store_analysis_result(
        self,
        job_id: str,
        patient_mrn: str,
        scan_id: str,
        tumor_grade: str,
        confidence: float,
        tumor_type: Optional[str],
        segmentation_results: list[dict],
        classification_details: Optional[dict] = None,
    ) -> None:
        """Store a complete analysis result in the graph (supports ensemble metadata)."""
        timestamp = datetime.utcnow().isoformat()

        models_loaded = 0
        agreement_score = 0.0
        if classification_details:
            models_loaded = classification_details.get("models_loaded", 0)
            agreement_score = classification_details.get("agreement_score", 0.0)

        # Ensure patient + scan exist
        self.graph.query("MERGE (p:Patient {mrn: $mrn})", {"mrn": patient_mrn})
        self.graph.query("MERGE (s:Scan {id: $scan_id}) SET s.date = $date", {"scan_id": scan_id, "date": timestamp})
        self.graph.query(
            "MATCH (p:Patient {mrn: $mrn}), (s:Scan {id: $scan_id}) MERGE (p)-[:HAS_SCAN]->(s)",
            {"mrn": patient_mrn, "scan_id": scan_id},
        )

        details_str = json.dumps(classification_details) if classification_details else ""
        self.graph.query(
            "CREATE (a:AnalysisResult {"
            "  job_id: $job_id, confidence: $confidence, grade: $grade, "
            "  timestamp: $ts, details: $details, "
            "  models_loaded: $models_loaded, agreement_score: $agreement"
            "})",
            {
                "job_id": job_id, "confidence": confidence, "grade": tumor_grade,
                "ts": timestamp, "details": details_str,
                "models_loaded": models_loaded, "agreement": agreement_score,
            },
        )

        self.graph.query(
            "MATCH (s:Scan {id: $scan_id}), (a:AnalysisResult {job_id: $job_id}) MERGE (s)-[:PRODUCED]->(a)",
            {"scan_id": scan_id, "job_id": job_id},
        )
        self.graph.query(
            "MATCH (a:AnalysisResult {job_id: $job_id}), (g:TumorGrade {grade: $grade}) MERGE (a)-[:CLASSIFIED_AS]->(g)",
            {"job_id": job_id, "grade": tumor_grade},
        )
        if tumor_type:
            self.graph.query(
                "MATCH (a:AnalysisResult {job_id: $job_id}), (t:TumorType {name: $ttype}) MERGE (a)-[:TUMOR_TYPE]->(t)",
                {"job_id": job_id, "ttype": tumor_type},
            )

        for seg in segmentation_results:
            self.graph.query(
                "CREATE (sr:SubregionResult {subregion: $subregion, confidence: $confidence, volume_mm3: $volume})",
                {"subregion": seg["subregion"], "confidence": seg["confidence"], "volume": seg.get("volume_mm3", 0)},
            )
            self.graph.query(
                "MATCH (a:AnalysisResult {job_id: $job_id}), "
                "      (sr:SubregionResult {subregion: $subregion, confidence: $confidence}) "
                "MERGE (a)-[:HAS_SEGMENTATION]->(sr)",
                {"job_id": job_id, "subregion": seg["subregion"], "confidence": seg["confidence"]},
            )

        logger.info(f"Stored analysis result in FalkorDB for job {job_id}")

    # ── Dataset / Training ────────────────────────────────────────

    def store_dataset_metadata(
        self,
        images: list[dict],
        source_dataset: str,
    ) -> None:
        """
        Store dataset image metadata in the graph.

        Args:
            images: List of dicts with {path, class_label, split}
            source_dataset: Name of the source dataset
        """
        batch_size = 500
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            for img in batch:
                self.graph.query(
                    "MERGE (d:DatasetImage {path: $path}) "
                    "SET d.class_label = $label, d.split = $split, "
                    "    d.source_dataset = $source",
                    {
                        "path": img["path"],
                        "label": img["class_label"],
                        "split": img["split"],
                        "source": source_dataset,
                    },
                )

        logger.info(f"Stored {len(images)} dataset images in FalkorDB")

    def store_training_run(
        self,
        run_id: str,
        accuracy: float,
        epochs: int,
        model_path: str,
        class_distribution: dict,
    ) -> None:
        """Store training run metadata."""
        self.graph.query(
            "CREATE (t:TrainingRun {"
            "  id: $id, accuracy: $accuracy, epochs: $epochs, "
            "  model_path: $model_path, timestamp: $ts, "
            "  class_distribution: $dist"
            "})",
            {
                "id": run_id,
                "accuracy": accuracy,
                "epochs": epochs,
                "model_path": model_path,
                "ts": datetime.utcnow().isoformat(),
                "dist": str(class_distribution),
            },
        )
        logger.info(f"Stored training run {run_id} in FalkorDB (acc={accuracy:.4f})")

    def get_analysis_history(self, patient_mrn: str) -> list[dict]:
        """Get all analysis results for a patient."""
        result = self.graph.query(
            "MATCH (p:Patient {mrn: $mrn})-[:HAS_SCAN]->(s)-[:PRODUCED]->(a) "
            "OPTIONAL MATCH (a)-[:CLASSIFIED_AS]->(g:TumorGrade) "
            "OPTIONAL MATCH (a)-[:TUMOR_TYPE]->(t:TumorType) "
            "RETURN a.job_id AS job_id, a.confidence AS confidence, "
            "       g.grade AS grade, t.name AS tumor_type, "
            "       a.timestamp AS timestamp "
            "ORDER BY a.timestamp DESC",
            {"mrn": patient_mrn},
        )
        return [
            {
                "job_id": row[0],
                "confidence": row[1],
                "grade": row[2],
                "tumor_type": row[3],
                "timestamp": row[4],
            }
            for row in result.result_set
        ]

    def get_grade_statistics(self) -> dict:
        """Get aggregate statistics across all analyses."""
        result = self.graph.query(
            "MATCH (a:AnalysisResult)-[:CLASSIFIED_AS]->(g:TumorGrade) "
            "RETURN g.grade AS grade, COUNT(a) AS count, AVG(a.confidence) AS avg_confidence "
            "ORDER BY count DESC"
        )
        return {
            row[0]: {"count": row[1], "avg_confidence": row[2]}
            for row in result.result_set
        }

    def get_dataset_overview(self) -> dict:
        """Get dataset statistics from the graph."""
        result = self.graph.query(
            "MATCH (d:DatasetImage) "
            "RETURN d.class_label AS label, d.split AS split, COUNT(d) AS count "
            "ORDER BY label, split"
        )
        overview = {}
        for row in result.result_set:
            label, split, count = row[0], row[1], row[2]
            if label not in overview:
                overview[label] = {}
            overview[label][split] = count
        return overview

    def get_training_history(self) -> list[dict]:
        """Get all training runs ordered by accuracy."""
        result = self.graph.query(
            "MATCH (t:TrainingRun) "
            "RETURN t.id, t.accuracy, t.epochs, t.model_path, t.timestamp "
            "ORDER BY t.accuracy DESC"
        )
        return [
            {
                "id": row[0],
                "accuracy": row[1],
                "epochs": row[2],
                "model_path": row[3],
                "timestamp": row[4],
            }
            for row in result.result_set
        ]

    def find_similar_cases(self, tumor_grade: str, min_confidence: float = 0.8) -> list[dict]:
        """Find historical cases with the same grade and high confidence."""
        result = self.graph.query(
            "MATCH (a:AnalysisResult)-[:CLASSIFIED_AS]->(g:TumorGrade {grade: $grade}) "
            "WHERE a.confidence >= $min_conf "
            "OPTIONAL MATCH (a)-[:HAS_SEGMENTATION]->(sr) "
            "RETURN a.job_id, a.confidence, a.timestamp, "
            "       COLLECT(sr.subregion) AS subregions, "
            "       COLLECT(sr.volume_mm3) AS volumes "
            "ORDER BY a.confidence DESC LIMIT 10",
            {"grade": tumor_grade, "min_conf": min_confidence},
        )
        return [
            {
                "job_id": row[0],
                "confidence": row[1],
                "timestamp": row[2],
                "subregions": row[3],
                "volumes": row[4],
            }
            for row in result.result_set
        ]

    def ping(self) -> bool:
        """Check if FalkorDB is reachable."""
        try:
            self.graph.query("RETURN 1")
            return True
        except Exception:
            return False

    # ── Doctor CRUD (M:N with Patient and InferenceJob) ───────────

    def create_doctor(self, doctor_id: str, name: str, specialization: str,
                      license_number: str, email: str) -> dict:
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "MERGE (d:Doctor {id: $id}) "
            "ON CREATE SET d.name = $name, d.specialization = $spec, "
            "  d.license_number = $lic, d.email = $email, d.created_at = $now "
            "ON MATCH SET d.name = $name, d.specialization = $spec, "
            "  d.license_number = $lic, d.email = $email",
            {"id": doctor_id, "name": name, "spec": specialization,
             "lic": license_number, "email": email, "now": now},
        )
        return {"id": doctor_id, "name": name, "specialization": specialization}

    def get_doctor(self, doctor_id: str) -> dict | None:
        result = self.graph.query(
            "MATCH (d:Doctor {id: $id}) "
            "RETURN d.id, d.name, d.specialization, d.license_number, d.email, d.created_at",
            {"id": doctor_id},
        )
        if not result.result_set:
            return None
        r = result.result_set[0]
        return {"id": r[0], "name": r[1], "specialization": r[2],
                "license_number": r[3], "email": r[4], "created_at": r[5]}

    def assign_doctor_to_patient(self, doctor_id: str, patient_mrn: str,
                                 notes: str = "") -> None:
        """M:N — A doctor can review many patients, a patient can be reviewed by many doctors."""
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "MATCH (d:Doctor {id: $did}), (p:Patient {mrn: $mrn}) "
            "MERGE (d)-[r:REVIEWED]->(p) "
            "SET r.reviewed_at = $now, r.notes = $notes, r.decision = 'pending'",
            {"did": doctor_id, "mrn": patient_mrn, "notes": notes, "now": now},
        )

    def assign_doctor_to_job(self, doctor_id: str, job_id: str,
                             priority: str = "normal") -> None:
        """M:N — A doctor can be assigned to many jobs, a job can have many reviewers."""
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "MATCH (d:Doctor {id: $did}), (j:InferenceJob {id: $jid}) "
            "MERGE (d)-[r:ASSIGNED_TO]->(j) "
            "SET r.assigned_at = $now, r.priority = $priority, r.status = 'pending'",
            {"did": doctor_id, "jid": job_id, "priority": priority, "now": now},
        )

    def get_doctor_patients(self, doctor_id: str) -> list[dict]:
        """Get all patients reviewed by a specific doctor."""
        result = self.graph.query(
            "MATCH (d:Doctor {id: $did})-[r:REVIEWED]->(p:Patient) "
            "RETURN p.mrn, r.reviewed_at, r.notes, r.decision "
            "ORDER BY r.reviewed_at DESC",
            {"did": doctor_id},
        )
        return [{"mrn": r[0], "reviewed_at": r[1], "notes": r[2], "decision": r[3]}
                for r in result.result_set]

    def get_patient_doctors(self, patient_mrn: str) -> list[dict]:
        """Get all doctors who have reviewed a specific patient."""
        result = self.graph.query(
            "MATCH (d:Doctor)-[r:REVIEWED]->(p:Patient {mrn: $mrn}) "
            "RETURN d.id, d.name, d.specialization, r.reviewed_at, r.decision "
            "ORDER BY r.reviewed_at DESC",
            {"mrn": patient_mrn},
        )
        return [{"id": r[0], "name": r[1], "specialization": r[2],
                 "reviewed_at": r[3], "decision": r[4]}
                for r in result.result_set]

    # ── Tags (M:N with Scans) ────────────────────────────────────

    def tag_scan(self, scan_id: str, tag_name: str) -> None:
        """M:N — A scan can have many tags, a tag can apply to many scans."""
        self.graph.query(
            "MERGE (tag:Tag {name: $name})",
            {"name": tag_name},
        )
        self.graph.query(
            "MATCH (s:Scan {id: $sid}), (tag:Tag {name: $name}) "
            "MERGE (s)-[:TAGGED_WITH]->(tag)",
            {"sid": scan_id, "name": tag_name},
        )

    def get_scans_by_tag(self, tag_name: str) -> list[dict]:
        result = self.graph.query(
            "MATCH (s:Scan)-[:TAGGED_WITH]->(tag:Tag {name: $name}) "
            "RETURN s.id, s.date, s.status",
            {"name": tag_name},
        )
        return [{"id": r[0], "date": r[1], "status": r[2]} for r in result.result_set]

    # ── Audit Log ─────────────────────────────────────────────────

    def create_audit_log(self, action: str, entity_type: str, entity_id: str,
                         actor: str, details: str = "") -> None:
        log_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "CREATE (al:AuditLog {"
            "  id: $id, action: $action, entity_type: $etype, entity_id: $eid, "
            "  actor: $actor, timestamp: $ts, details: $details})",
            {"id": log_id, "action": action, "etype": entity_type,
             "eid": entity_id, "actor": actor, "ts": now, "details": details},
        )

    def get_audit_logs(self, entity_type: str | None = None, limit: int = 50) -> list[dict]:
        if entity_type:
            result = self.graph.query(
                "MATCH (al:AuditLog {entity_type: $etype}) "
                "RETURN al.id, al.action, al.entity_type, al.entity_id, "
                "       al.actor, al.timestamp, al.details "
                "ORDER BY al.timestamp DESC LIMIT $limit",
                {"etype": entity_type, "limit": limit},
            )
        else:
            result = self.graph.query(
                "MATCH (al:AuditLog) "
                "RETURN al.id, al.action, al.entity_type, al.entity_id, "
                "       al.actor, al.timestamp, al.details "
                "ORDER BY al.timestamp DESC LIMIT $limit",
                {"limit": limit},
            )
        return [{"id": r[0], "action": r[1], "entity_type": r[2], "entity_id": r[3],
                 "actor": r[4], "timestamp": r[5], "details": r[6]}
                for r in result.result_set]

    # ── Model Versioning ──────────────────────────────────────────

    def create_model_version(self, model_name: str, version: str,
                             accuracy: float, path: str) -> str:
        version_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        self.graph.query(
            "CREATE (mv:ModelVersion {"
            "  id: $id, model_name: $name, version: $ver, accuracy: $acc, "
            "  path: $path, created_at: $now, status: 'active'})",
            {"id": version_id, "name": model_name, "ver": version,
             "acc": accuracy, "path": path, "now": now},
        )
        # Link to previous version (SUPERSEDES chain)
        self.graph.query(
            "MATCH (new:ModelVersion {id: $new_id}), "
            "      (old:ModelVersion {model_name: $name, status: 'active'}) "
            "WHERE old.id <> $new_id "
            "SET old.status = 'superseded' "
            "MERGE (new)-[:SUPERSEDES]->(old)",
            {"new_id": version_id, "name": model_name},
        )
        return version_id

    def get_model_versions(self, model_name: str) -> list[dict]:
        result = self.graph.query(
            "MATCH (mv:ModelVersion {model_name: $name}) "
            "RETURN mv.id, mv.version, mv.accuracy, mv.path, mv.status, mv.created_at "
            "ORDER BY mv.created_at DESC",
            {"name": model_name},
        )
        return [{"id": r[0], "version": r[1], "accuracy": r[2],
                 "path": r[3], "status": r[4], "created_at": r[5]}
                for r in result.result_set]

    def get_active_model_version(self, model_name: str) -> dict | None:
        result = self.graph.query(
            "MATCH (mv:ModelVersion {model_name: $name, status: 'active'}) "
            "RETURN mv.id, mv.version, mv.accuracy, mv.path, mv.created_at "
            "ORDER BY mv.created_at DESC LIMIT 1",
            {"name": model_name},
        )
        if not result.result_set:
            return None
        r = result.result_set[0]
        return {"id": r[0], "version": r[1], "accuracy": r[2], "path": r[3], "created_at": r[4]}


# Singleton
_falkordb_instance: Optional[FalkorDBService] = None


def get_falkordb() -> FalkorDBService:
    """Get the FalkorDB service singleton."""
    global _falkordb_instance
    if _falkordb_instance is None:
        _falkordb_instance = FalkorDBService()
    return _falkordb_instance
