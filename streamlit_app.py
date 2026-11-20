try:
    __import__("pysqlite3")
    import sys as _sys
    _sys.modules["sqlite3"] = _sys.modules.pop("pysqlite3")
except ImportError:
    pass

import base64
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

import chromadb
from google import genai
from google.genai import types
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pypdf import PdfReader
import streamlit as st

# ---------------------------------------------------------------------
# 1. KONFIGURASI JALUR BERKAS, MODEL FAILOVER, & KOORDINAT JAWA BARAT
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MASTER_CSV_PATH = BASE_DIR / "west_java_dengue_master_2016_2025.csv"
PDF_PATH = BASE_DIR / "Pedoman_DBD_Kemenkes_2017.pdf"
LOGO_PATH = BASE_DIR / "dengue_logo.png"

MODEL_CANDIDATES = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]
IR_THRESHOLD = 49.0
CFR_THRESHOLD = 1.0
APPROVER_ROLE = "Kepala Daerah (Bupati/Walikota)"

WEST_JAVA_COORDS = {
    "Kab.Bandung": (-7.025, 107.520),
    "Kab.Bandung Barat": (-6.845, 107.489),
    "Kab.Bekasi": (-6.265, 107.143),
    "Kab.Bogor": (-6.479, 106.825),
    "Kab.Ciamis": (-7.326, 108.353),
    "Kab.Cianjur": (-6.822, 107.140),
    "Kab.Cirebon": (-6.764, 108.479),
    "Kab.Garut": (-7.216, 107.901),
    "Kab.Indramayu": (-6.327, 108.325),
    "Kab.Karawang": (-6.305, 107.301),
    "Kab.Kuningan": (-6.976, 108.483),
    "Kab.Majalengka": (-6.836, 108.227),
    "Kab.Pangandaran": (-7.684, 108.650),
    "Kab.Purwakarta": (-6.556, 107.443),
    "Kab.Subang": (-6.571, 107.759),
    "Kab.Sukabumi": (-6.988, 106.551),
    "Kab.Sumedang": (-6.838, 107.928),
    "Kab.Tasikmalaya": (-7.359, 108.107),
    "Kota Bandung": (-6.917, 107.619),
    "Kota Banjar": (-7.370, 108.534),
    "Kota Bekasi": (-6.238, 106.976),
    "Kota Bogor": (-6.597, 106.806),
    "Kota Cimahi": (-6.872, 107.542),
    "Kota Cirebon": (-6.732, 108.552),
    "Kota Depok": (-6.402, 106.794),
    "Kota Sukabumi": (-6.923, 106.929),
    "Kota Tasikmalaya": (-7.327, 108.221),
}

FOCUS_MODES = [
    "1. Laporan Komprehensif Lengkap (Standar 5 Bagian)",
    "2. Fokus Spesifik: Epidemiologi & Audit Klinis (IR, CFR & Siklus)",
    "3. Fokus Spesifik: Korelasi Iklim, Banjir DIBI & Vektor",
    "4. Fokus Spesifik: Protokol Teknis & Dosis SOP Kemenkes",
    "5. Fokus Spesifik: Proyeksi & Kesiapsiagaan Tahun Berjalan (2026+)",
]

FOCUS_INSTRUCTIONS = {
    FOCUS_MODES[0]: (
        "Susun laporan komprehensif 5 bagian lengkap:\n"
        "### 1. RINGKASAN SITUASI EPIDEMIOLOGI\n"
        "### 2. ANALISIS DINAMIKA TREN 10 TAHUN (2016-2025) & DETEKSI SIKLUS\n"
        "### 3. ANALISIS FAKTOR RISIKO IKLIM DAN BENCANA BANJIR (DIBI BNPB)\n"
        "### 4. RUJUKAN REGULASI RESMI KEMENKES RI (KUTIP HALAMAN)\n"
        "### 5. DRAF TINDAKAN MITIGASI TERPADU & PROYEKSI KESIAPSIAGAAN (2026+)"
    ),
    FOCUS_MODES[1]: (
        "Pengguna memilih FOKUS SPESIFIK EPIDEMIOLOGI & AUDIT KLINIS. Jangan buat laporan umum yang melebar; "
        "fokuskan seluruh analisis secara mendalam hanya pada 3 bagian berikut:\n"
        "### 1. EVALUASI KETAT INCIDENCE RATE (IR) & LETALITAS (CFR) TERHADAP STANDAR NASIONAL\n"
        "### 2. BEDAH POLA SIKLUS WABAH 10 TAHUN (2016-2025) & DETEKSI ANOMALI LONJAKAN\n"
        "### 3. REKOMENDASI AUDIT KLINIS FASKES, TRIASE IGD, DAN SISTEM RUJUKAN RS (KUTIP HALAMAN SOP)"
    ),
    FOCUS_MODES[2]: (
        "Pengguna memilih FOKUS SPESIFIK KORELASI IKLIM, BANJIR DIBI & EKOLOGI VEKTOR. "
        "Fokuskan seluruh analisis secara tajam hanya pada 3 bagian berikut:\n"
        "### 1. KORELASI CURAH HUJAN TAHUNAN & KEJADIAN BANJIR DIBI BNPB TERHADAP PROLIFERASI AEDES\n"
        "### 2. PEMETAAN KERAWANAN GENANGAN & RUMAH TERENDAM SELAMA 10 TAHUN (2016-2025)\n"
        "### 3. INTERVENSI LINGKUNGAN PASCA-BANJIR, SANITASI KONTAINER, & LARVASIDASI SELEKTIF (KUTIP HALAMAN SOP)"
    ),
    FOCUS_MODES[3]: (
        "Pengguna memilih FOKUS SPESIFIK PROTOKOL TEKNIS & DOSIS LAPANGAN (SOP KEMENKES 2017). "
        "Fokuskan seluruh isi laporan pada panduan teknis operasional petugas lapangan:\n"
        "### 1. KRITERIA PENYELIDIKAN EPIDEMIOLOGI (PE RADIUS 100M) & PENANGGULANGAN FOKUS (RADIUS 200M)\n"
        "### 2. SPESIFIKASI TEKNIS FOGGING FOKUS (2 SIKLUS INTERVAL 7 HARI) & LARVASIDASI (KUTIP HALAMAN SOP)\n"
        "### 3. ALUR PELAPORAN BERJENJANG PUSKESMAS KE DINKES, CAMAT, DAN LURAH (HALAMAN 98)"
    ),
    FOCUS_MODES[4]: (
        "Pengguna memilih FOKUS SPESIFIK PROYEKSI & KESIAPSIAGAAN TAHUN BERJALAN (DI LUAR DATASET / 2026+). "
        "Gunakan baseline statistik 10 tahun (2016-2025) tanpa mengarang angka fiktif untuk menyusun:\n"
        "### 1. PROFIL BASELINE HISTORIS 10 TAHUN (2016-2025) & SIKLUS BERULANG WILAYAH\n"
        "### 2. PROYEKSI RISIKO TRANSMISI & KERAWANAN HIDROMETEOROLOGI TAHUN BERJALAN (2026+)\n"
        "### 3. RENCANA KONTINJENSI PRE-EMPTIF & SISTEM KEWASPADAAN DINI (SKD-KLB KEMENKES RI)"
    ),
}

FOCUS_RAG_KEYWORDS = {
    FOCUS_MODES[1]: " tatalaksana klinis triase syok dengue rujukan rumah sakit",
    FOCUS_MODES[2]: " siklus ekologis nyamuk aedes aegypti curah hujan genangan air kontainer",
    FOCUS_MODES[4]: " sistem kewaspadaan dini SKD-KLB pemantauan mingguan pra musim hujan",
}

INJECTION_PATTERNS = (
    "ignore previous", "ignore all", "abaikan instruksi", "lupakan aturan",
    "system prompt", "jailbreak", "bypass", "act as ", "pretend to be", "berpura-pura",
)
OFF_TOPIC_PATTERNS = (
    "resep ", "masakan", "sepak bola", "piala dunia", "saham", "bitcoin", "kripto",
    "crypto", "politik", "pemilu", "partai", "puisi", "cerpen", "novel", "lirik lagu",
    "kode python", "javascript", "html game", "horoskop", "zodiak",
)
DOMAIN_KEYWORDS = (
    "dbd", "dengue", "demam", "nyamuk", "aedes", "jentik", "larva", "larvasid", "abate",
    "fogging", "pengasapan", "psn", "3m", "ir", "incidence", "cfr", "fatality", "kematian",
    "kasus", "wabah", "klb", "skd", "puskesmas", "rs", "rumah sakit", "faskes", "triase",
    "syok", "trombosit", "kemenkes", "sop", "pedoman", "pe", "penyelidikan", "radius",
    "hujan", "curah", "banjir", "dibi", "bnpb", "genangan", "drainase", "selokan", "ban bekas",
    "kontainer", "bak mandi", "wilayah", "kecamatan", "kelurahan", "desa", "warga", "sekolah",
    "jawa barat", "jabar", "kab.", "kota ", "proyeksi", "kesiapsiagaan", "vektor", "mitigasi",
    "pasien", "rujukan", "masyarakat", "lingkungan", "kebersihan", "sampah", "air",
    "langkah", "tindakan", "dosis", "pelaporan", "camat", "lurah", "dinkes", "kesehatan",
    "obat", "rawat", "diagnos", "gejala", "panas", "muntah", "darah", "jumantik",
)


# ---------------------------------------------------------------------
# 2. UTILITAS JALUR TULIS & INISIALISASI CACHED DATA / RAG / GEMINI
# ---------------------------------------------------------------------
def is_dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.touch()
        probe.unlink()
        return True
    except Exception:
        return False


def resolve_runtime_paths():
    local_chroma = BASE_DIR / "chroma_kemenkes_db"
    if local_chroma.exists() and is_dir_writable(local_chroma):
        chroma_path = local_chroma
    else:
        chroma_path = Path(tempfile.gettempdir()) / "dengueguard_chroma_db"
        chroma_path.mkdir(parents=True, exist_ok=True)

    if is_dir_writable(BASE_DIR):
        audit_path = str(BASE_DIR / "audit_log.jsonl")
    else:
        audit_path = str(Path(tempfile.gettempdir()) / "dengueguard_audit_log.jsonl")
    return chroma_path, audit_path


CHROMA_DIR, AUDIT_LOG_FILE = resolve_runtime_paths()


@st.cache_data
def load_master_dataset():
    return pd.read_csv(MASTER_CSV_PATH)


def split_text(text, chunk_size=800, chunk_overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


@st.cache_resource
def init_kemenkes_collection():
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col_name = "kemenkes_dengue_sop"
    try:
        col = chroma_client.get_or_create_collection(name=col_name)
        if col.count() >= 300:
            return col
    except Exception:
        pass

    reader = PdfReader(str(PDF_PATH))
    all_chunks, all_metadatas, all_ids = [], [], []
    counter = 1
    for page_idx, page in enumerate(reader.pages, start=1):
        txt = (page.extract_text() or "").strip()
        if len(txt) <= 50:
            continue
        for chunk in split_text(txt):
            all_chunks.append(chunk)
            all_metadatas.append({"page_number": page_idx})
            all_ids.append(f"chunk_{counter}")
            counter += 1

    try:
        chroma_client.delete_collection(name=col_name)
    except Exception:
        pass
    col = chroma_client.create_collection(name=col_name)
    for i in range(0, len(all_chunks), 100):
        col.add(
            documents=all_chunks[i:i + 100],
            metadatas=all_metadatas[i:i + 100],
            ids=all_ids[i:i + 100],
        )
    return col


def _read_secret_api_key():
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        try:
            val = str(st.secrets.get(key_name, "")).strip()
            if val:
                return val
        except Exception:
            pass
    env_val = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if env_val:
        return env_val
    for key_file in (BASE_DIR / "Gemini API.txt", BASE_DIR.parent.parent / "Gemini API.txt"):
        if key_file.exists():
            txt = key_file.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    return ""


@st.cache_resource
def get_gemini_client():
    api_key = _read_secret_api_key()
    return genai.Client(api_key=api_key) if api_key else None


# ---------------------------------------------------------------------
# 3. EMPAT ALAT AGENTIC TERSTANDARISASI (MCP TOOLS JAWA BARAT)
# ---------------------------------------------------------------------
def tool_get_epidemiology(region: str, year: int) -> str:
    """Mengambil data insidensi DBD (IR) dan tingkat kematian (CFR) wilayah Jawa Barat."""
    df = load_master_dataset()
    reg_df = df[df["regency_city"].str.lower() == region.lower()].sort_values("year")
    if reg_df.empty:
        return json.dumps({"status": "error", "message": f"Wilayah {region} tidak ditemukan di Jawa Barat."})
    data = reg_df[reg_df["year"] == int(year)]
    if data.empty:
        latest = reg_df.iloc[-1]
        return json.dumps({
            "status": "out_of_dataset_projection",
            "input_contract": {"region": region, "requested_year": int(year)},
            "region": latest["regency_city"],
            "note": f"Tahun {year} di luar dataset historis (2016-2025). Menggunakan baseline 10 tahun.",
            "baseline_mean_ir_10y": round(float(reg_df["dengue_incidence_rate"].mean()), 2),
            "baseline_mean_cfr_10y": round(float(reg_df["dengue_case_fatality_rate"].mean()), 2),
            "latest_year_recorded": int(latest["year"]),
            "latest_incidence_rate": float(latest["dengue_incidence_rate"]),
            "latest_case_fatality_rate": float(latest["dengue_case_fatality_rate"]),
        }, indent=2)
    row = data.iloc[0]
    return json.dumps({
        "status": "success",
        "input_contract": {"region": row["regency_city"], "year": int(row["year"])},
        "region": row["regency_city"],
        "year": int(row["year"]),
        "incidence_rate": float(row["dengue_incidence_rate"]),
        "case_fatality_rate": float(row["dengue_case_fatality_rate"]),
    }, indent=2)


def tool_get_climate_flood(region: str, year: int) -> str:
    """Mengambil metrik curah hujan tahunan (mm), kejadian banjir, dan rumah terendam (DIBI BNPB)."""
    df = load_master_dataset()
    reg_df = df[df["regency_city"].str.lower() == region.lower()].sort_values("year")
    if reg_df.empty:
        return json.dumps({"status": "error", "message": f"Wilayah {region} tidak ditemukan di Jawa Barat."})
    data = reg_df[reg_df["year"] == int(year)]
    if data.empty:
        latest = reg_df.iloc[-1]
        return json.dumps({
            "status": "out_of_dataset_projection",
            "input_contract": {"region": region, "requested_year": int(year)},
            "region": latest["regency_city"],
            "note": f"Tahun {year} di luar dataset (2016-2025). Menggunakan rata-rata 10 tahun.",
            "avg_annual_rainfall_10y_mm": round(float(reg_df["annual_rainfall_mm"].mean()), 1),
            "avg_flood_events_10y": round(float(reg_df["flood_events_count"].mean()), 1),
            "total_flooded_houses_10y": int(reg_df["flooded_houses_count"].sum()),
            "latest_year_rainfall_mm": float(latest["annual_rainfall_mm"]),
            "latest_year_flood_events": int(latest["flood_events_count"]),
            "latest_year_flooded_houses": int(latest["flooded_houses_count"]),
        }, indent=2)
    row = data.iloc[0]
    return json.dumps({
        "status": "success",
        "input_contract": {"region": row["regency_city"], "year": int(row["year"])},
        "region": row["regency_city"],
        "year": int(row["year"]),
        "annual_rainfall_mm": float(row["annual_rainfall_mm"]),
        "flood_events_count": int(row["flood_events_count"]),
        "flooded_houses_count": int(row["flooded_houses_count"]),
    }, indent=2)


def tool_get_historical_trend(region: str) -> str:
    """Mengambil deret waktu 10 tahun (2016-2025) beserta tahun puncak outbreak."""
    df = load_master_dataset()
    data = df[df["regency_city"].str.lower() == region.lower()].sort_values("year")
    if data.empty:
        return json.dumps({"status": "error", "message": f"Data tren historis tidak ditemukan untuk {region}."})
    trend_records = [
        {
            "year": int(row["year"]),
            "incidence_rate": float(row["dengue_incidence_rate"]),
            "case_fatality_rate": float(row["dengue_case_fatality_rate"]),
            "annual_rainfall_mm": float(row["annual_rainfall_mm"]),
            "flood_events": int(row["flood_events_count"]),
            "flooded_houses": int(row["flooded_houses_count"]),
        }
        for _, row in data.iterrows()
    ]
    peak_row = data.loc[data["dengue_incidence_rate"].idxmax()]
    return json.dumps({
        "status": "success",
        "input_contract": {"region": data.iloc[0]["regency_city"], "period": "2016-2025 (10 Years)"},
        "region": data.iloc[0]["regency_city"],
        "records_count": len(trend_records),
        "summary_10y": {
            "mean_ir_10y": round(float(data["dengue_incidence_rate"].mean()), 2),
            "mean_cfr_10y": round(float(data["dengue_case_fatality_rate"].mean()), 2),
            "peak_outbreak_year": int(peak_row["year"]),
            "peak_outbreak_ir": float(peak_row["dengue_incidence_rate"]),
            "total_flood_events_10y": int(data["flood_events_count"].sum()),
            "total_flooded_houses_10y": int(data["flooded_houses_count"].sum()),
        },
        "history": trend_records,
    }, indent=2)


def tool_query_kemenkes_sop(query: str, n_results: int = 3) -> str:
    """Melakukan pencarian semantik pada basis pengetahuan Pedoman DBD Kemenkes RI 2017."""
    col = init_kemenkes_collection()
    search_results = col.query(query_texts=[query], n_results=n_results)
    retrieved_excerpts = [
        {
            "source": f"Pedoman DBD Kemenkes 2017 (Halaman {search_results['metadatas'][0][i]['page_number']})",
            "content": search_results["documents"][0][i],
        }
        for i in range(len(search_results["documents"][0]))
    ]
    return json.dumps({
        "status": "success",
        "input_contract": {"query": query, "n_results": int(n_results)},
        "results": retrieved_excerpts,
    }, indent=2)


# ---------------------------------------------------------------------
# 4. GUARDRAIL TOPIK 2 LAPIS, AUDIT TRAIL SHA-256, & HITL STATE GUARD
# ---------------------------------------------------------------------
def check_topic_guardrail(text: str):
    """Memastikan masukan natural tetap pada topik DBD, iklim-banjir, vektor, atau Kemenkes."""
    clean = (text or "").strip().lower()
    if not clean:
        return True, "", ""
    if any(p in clean for p in INJECTION_PATTERNS):
        return (
            False,
            "INJECTION_BLOCKED",
            "Instruksi ditolak: Sistem mendeteksi indikasi pengubahan peran/aturan agen (Prompt Injection).",
        )
    if any(p in clean for p in OFF_TOPIC_PATTERNS):
        return (
            False,
            "OUT_OF_SCOPE_BLOCKED",
            "Topik di luar cakupan sistem: Pertanyaan hanya boleh membahas epidemiologi DBD, iklim/banjir Jawa Barat, vektor Aedes aegypti, atau SOP Kemenkes RI.",
        )
    if len(clean) >= 12 and not any(k in clean for k in DOMAIN_KEYWORDS):
        return (
            False,
            "OUT_OF_SCOPE_BLOCKED",
            "Pertanyaan di luar topik DBD/Kesehatan Masyarakat: Sertakan konteks kasus DBD, curah hujan/banjir, nyamuk/jentik, faskes, atau pedoman Kemenkes.",
        )
    return True, "", ""


class AuditLogger:
    def __init__(self, filepath=AUDIT_LOG_FILE):
        self.filepath = filepath
        self.last_hash = "GENESIS"
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.last_hash = json.loads(line).get("record_hash", self.last_hash)
            except Exception:
                self.last_hash = "GENESIS"

    def log(self, actor, action, plan_id, details=""):
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = f"{ts}|{actor}|{action}|{plan_id}|{details}|{self.last_hash}"
        record_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        entry = {
            "timestamp": ts,
            "actor": actor,
            "action": action,
            "plan_id": plan_id,
            "details": details,
            "prev_hash": self.last_hash,
            "record_hash": record_hash,
        }
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.last_hash = record_hash
        return entry

    def verify(self):
        if not os.path.exists(self.filepath):
            return "<div class='msg-box msg-warn'>Berkas audit log belum ditemukan.</div>"
        prev = "GENESIS"
        idx = 0
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                idx += 1
                rec = json.loads(line)
                payload = f"{rec['timestamp']}|{rec['actor']}|{rec['action']}|{rec['plan_id']}|{rec['details']}|{prev}"
                expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                if rec["prev_hash"] != prev or rec["record_hash"] != expected:
                    return (
                        f"<div class='msg-box msg-err'>"
                        f"<strong>INTEGRITAS RUSAK:</strong> Terdeteksi ketidakcocokan rantai hash pada baris ke-{idx}."
                        f"</div>"
                    )
                prev = rec["record_hash"]
        return (
            f"<div class='msg-box msg-ok'>"
            f"<strong>INTEGRITAS TERVERIFIKASI:</strong> Seluruh rantai hash ({idx} entri log) valid dan otentik."
            f"</div>"
        )

    def table(self):
        if not os.path.exists(self.filepath):
            return pd.DataFrame(columns=["Waktu (UTC)", "Aktor", "Aksi", "ID Rencana", "Rincian", "Hash SHA-256"])
        rows = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append({
                        "Waktu (UTC)": r["timestamp"][:19].replace("T", " "),
                        "Aktor": r["actor"],
                        "Aksi": r["action"],
                        "ID Rencana": r["plan_id"],
                        "Rincian": r["details"][:55],
                        "Hash SHA-256": r["record_hash"][:14] + "...",
                    })
        return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def get_audit_logger():
    return AuditLogger(AUDIT_LOG_FILE)


def new_plan(region, year, report_text, source="Agentic-Gemini", focus_mode=""):
    clean_reg = region.replace("Kab.", "").replace("Kota ", "")[:3].upper()
    return {
        "plan_id": f"PLAN-{clean_reg}-{year}-{uuid.uuid4().hex[:6].upper()}",
        "region": region,
        "year": int(year),
        "focus_mode": focus_mode,
        "status": "PENDING",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "approved_by": None,
        "rejection_reason": None,
        "report": report_text,
        "source": source,
    }


def validate_pending_transition(plan, actor_name, actor_role):
    if plan is None:
        raise ValueError("Belum ada draf rencana aktif. Analis Dinkes harus menjalankan investigasi terlebih dahulu.")
    if plan["status"] != "PENDING":
        raise ValueError(f"Hanya rencana berstatus PENDING yang dapat diproses (Status saat ini: {plan['status']}).")
    if not actor_name.strip():
        raise ValueError("Nama pejabat otorisasi wajib diisi.")
    if actor_role != APPROVER_ROLE:
        raise ValueError(f"Akses ditolak: Hanya peran {APPROVER_ROLE} yang berwenang melakukan otorisasi ini.")


def approve_plan(plan, actor_name, actor_role, logger):
    validate_pending_transition(plan, actor_name, actor_role)
    plan = dict(plan, status="APPROVED", approved_by=actor_name.strip())
    logger.log(actor_name.strip(), "APPROVE", plan["plan_id"], f"Disetujui oleh {actor_role}")
    return plan


def reject_plan(plan, actor_name, actor_role, reason, logger):
    validate_pending_transition(plan, actor_name, actor_role)
    if not reason.strip():
        raise ValueError("Alasan penolakan wajib dicantumkan untuk kelengkapan audit.")
    plan = dict(plan, status="REJECTED", rejection_reason=reason.strip())
    logger.log(actor_name.strip(), "REJECT", plan["plan_id"], f"Alasan: {reason.strip()}")
    return plan


def dispatch_plan(plan, actor_name, logger):
    if plan is None or plan["status"] != "APPROVED":
        raise ValueError("Instruksi lapangan hanya dapat dikirim setelah draf disetujui (APPROVED) oleh Kepala Daerah.")
    plan = dict(plan, status="DISPATCHED")
    logger.log(actor_name.strip() or "Petugas Dinkes", "DISPATCH", plan["plan_id"], "Instruksi dikirim ke Puskesmas dan tim lapangan")
    return plan


def execute_plan(plan, actor_name, logger):
    if plan is None or plan["status"] != "DISPATCHED":
        raise ValueError("Hanya rencana berstatus DISPATCHED yang dapat ditandai selesai dilaksanakan (EXECUTED).")
    plan = dict(plan, status="EXECUTED")
    logger.log(actor_name.strip() or "Petugas Dinkes", "EXECUTE", plan["plan_id"], "Intervensi pengendalian vektor selesai dilaksanakan")
    return plan


# ---------------------------------------------------------------------
# 5. KOMPONEN HTML & VISUALISASI PLOTLY
# ---------------------------------------------------------------------
def parse_year_selection(year_str):
    parts = str(year_str).split()
    return int(parts[0]) if parts and parts[0].isdigit() else 2024


@st.cache_data
def load_logo_html():
    if not LOGO_PATH.exists():
        return ""
    b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("utf-8")
    return (
        f'<div class="header-logo-wrap">'
        f'<img src="data:image/png;base64,{b64}" alt="Logo DengueGuard" class="header-logo-img"/>'
        f'</div>'
    )


def render_intro_glossary_html():
    return """
    <div class="intro-box">
        <div class="intro-grid">
            <div class="intro-col">
                <div class="intro-kicker">PENGANTAR &amp; CARA PENGGUNAAN SISTEM</div>
                <h3 class="intro-heading">Selamat Datang di DengueGuard-Agent (Jawa Barat)</h3>
                <p class="intro-p">
                    <strong>DengueGuard-Agent</strong> adalah sistem pendukung keputusan berbasis <em>Agentic AI</em>
                    dan <em>Retrieval-Augmented Generation</em> (RAG) untuk analisis risiko serta perencanaan mitigasi
                    Demam Berdarah Dengue (DBD) pada 27 Kabupaten/Kota di Provinsi Jawa Barat (2016–2025 serta proyeksi 2026+).
                </p>
                <div class="workflow-steps">
                    <div class="wf-step"><strong>1. Peran Analis Dinas Kesehatan (P2P):</strong> Memilih wilayah &amp; tahun, mengajukan pertanyaan/catatan lapangan, menjalankan 4 alat analisis &amp; RAG Kemenkes untuk menghasilkan draf rencana (<code>PENDING</code>), lalu menindaklanjuti lapangan setelah disetujui.</div>
                    <div class="wf-step"><strong>2. Peran Kepala Daerah (Bupati/Walikota):</strong> Meninjau draf rekomendasi dan peta risiko provinsi, lalu memberikan otorisasi eksekutif resmi (<strong>Approve</strong> atau <strong>Reject</strong>).</div>
                    <div class="wf-step"><strong>3. Audit Kriptografis SHA-256:</strong> Setiap kueri agen, pemblokiran topik di luar domain, persetujuan, dan pengiriman perintah tercatat permanen untuk menjamin akuntabilitas.</div>
                </div>
            </div>
            <div class="glossary-col">
                <div class="intro-kicker">GLOSARIUM ISTILAH &amp; INDIKATOR KUNCI</div>
                <dl class="glossary-list">
                    <dt>IR (Incidence Rate)</dt>
                    <dd>Angka kejadian kasus baru DBD per 100.000 penduduk per tahun. Ambang batas waspada nasional Kemenkes adalah <strong>&le; 49,0 / 100rb</strong>.</dd>
                    <dt>CFR (Case Fatality Rate)</dt>
                    <dd>Persentase kematian penderita terhadap total kasus DBD terkonfirmasi. Target maksimal nasional Kemenkes adalah <strong>&lt; 1,0%</strong>.</dd>
                    <dt>DIBI BNPB &amp; Open-Meteo</dt>
                    <dd>Basis Data Informasi Bencana Indonesia (DIBI) BNPB untuk kejadian banjir &amp; rumah terendam, dipadukan data curah hujan tahunan (mm) dari Open-Meteo.</dd>
                    <dt>PE &amp; Fogging Fokus (SOP 2017)</dt>
                    <dd>Penyelidikan Epidemiologi radius 100 m (Hal. 94) serta PSN 3M Plus, larvasidasi, dan pengasapan fokus 2 siklus radius 200 m (Hal. 99).</dd>
                    <dt>MCP Tools, RAG &amp; HITL</dt>
                    <dd>Alat data terstandar (MCP-style), pencarian semantik Pedoman Kemenkes 2017 (RAG ChromaDB), dan gerbang persetujuan manusia (<em>Human-in-the-Loop</em>).</dd>
                </dl>
            </div>
        </div>
    </div>
    """


def render_footer_html():
    return """
    <div class="footer-box">
        <div><strong>&copy; 2026 DengueGuard-Agent</strong> &mdash; West Java Dengue Mitigation Decision Support System.</div>
        <div>Developed by <strong>Muhamad Farhan Budiana</strong> &amp; <strong>Nadya Ayu Lestari</strong> &bull; Department of Electrical Engineering, Universitas Indonesia.</div>
        <div class="footer-rights">All Rights Reserved.</div>
    </div>
    """


def render_status_bar(plan):
    if plan is None:
        return """
        <div class="status-banner">
            <div>
                <span class="status-label">STATUS RENCANA AKTIF</span>
                <div class="status-main">Belum Ada Rencana Dianalisis</div>
            </div>
            <div class="status-meta">Analis Dinkes memilih wilayah &amp; tahun pada Tab 1, lalu menjalankan Investigasi Agentic AI.</div>
        </div>
        """
    st_val = plan["status"]
    badge_cls = {
        "PENDING": "badge-pending",
        "APPROVED": "badge-approved",
        "DISPATCHED": "badge-dispatched",
        "EXECUTED": "badge-executed",
        "REJECTED": "badge-rejected",
    }.get(st_val, "badge-pending")

    extra_info = f"Mesin Inferensi: <strong>{plan.get('source', '-')}</strong>"
    if plan.get("focus_mode"):
        extra_info += f" &nbsp;|&nbsp; Fokus: <strong>{plan['focus_mode']}</strong>"
    if plan.get("approved_by"):
        extra_info += f" &nbsp;|&nbsp; Pejabat Otorisasi: <strong>{plan['approved_by']}</strong>"
    if plan.get("rejection_reason"):
        extra_info += f" &nbsp;|&nbsp; Alasan Penolakan: <strong>{plan['rejection_reason']}</strong>"

    return f"""
    <div class="status-banner">
        <div>
            <span class="status-label">ID RENCANA: {plan['plan_id']} &nbsp;&bull;&nbsp; WILAYAH: {plan['region'].upper()} ({plan['year']})</span>
            <div class="status-main">{extra_info}</div>
        </div>
        <div><span class="status-pill {badge_cls}">STATUS: {st_val}</span></div>
    </div>
    """


def render_plan_preview_md(plan):
    if plan is None:
        return (
            "### Belum Ada Draf Rencana Aktif\n"
            "Belum terdapat usulan mitigasi yang perlu ditinjau atau ditindaklanjuti. "
            "Silakan masuk melalui peran **Analis Dinas Kesehatan (P2P)** untuk menjalankan analisis wilayah terlebih dahulu."
        )
    return (
        f"### Ringkasan Draf Rencana Aktif: `{plan['plan_id']}`\n"
        f"- **Wilayah Sasaran:** {plan['region']} (Tahun Evaluasi/Proyeksi: **{plan['year']}**)\n"
        f"- **Status Saat Ini:** **{plan['status']}**\n"
        f"- **Sumber Inferensi:** {plan.get('source', '-')}\n\n"
        f"---\n\n{plan['report']}"
    )


def render_projection_kpi_cards(year, trend_data):
    mean_ir = float(trend_data["dengue_incidence_rate"].mean())
    mean_cfr = float(trend_data["dengue_case_fatality_rate"].mean())
    mean_rain = float(trend_data["annual_rainfall_mm"].mean())
    tot_flood = int(trend_data["flood_events_count"].sum())
    tot_houses = int(trend_data["flooded_houses_count"].sum())
    peak_row = trend_data.loc[trend_data["dengue_incidence_rate"].idxmax()]
    ir_alert = "kpi-alert" if mean_ir > IR_THRESHOLD else "kpi-normal"
    cfr_alert = "kpi-alert" if mean_cfr > CFR_THRESHOLD else "kpi-normal"
    return f"""
    <div class="kpi-grid">
        <div class="kpi-card {ir_alert}">
            <div class="kpi-title">BASELINE IR 10 TAHUN (PROYEKSI {year})</div>
            <div class="kpi-value">{mean_ir:.2f} <span class="kpi-unit">/ 100rb</span></div>
            <div class="kpi-sub">Puncak Siklus: {peak_row['dengue_incidence_rate']:.2f} ({int(peak_row['year'])})</div>
        </div>
        <div class="kpi-card {cfr_alert}">
            <div class="kpi-title">BASELINE CFR 10 TAHUN (PROYEKSI {year})</div>
            <div class="kpi-value">{mean_cfr:.2f}%</div>
            <div class="kpi-sub">Rata-rata 2016-2025 (Target &lt; {CFR_THRESHOLD}%)</div>
        </div>
        <div class="kpi-card kpi-neutral">
            <div class="kpi-title">RATA-RATA CURAH HUJAN 10 THN</div>
            <div class="kpi-value">{mean_rain:,.1f} <span class="kpi-unit">mm</span></div>
            <div class="kpi-sub">Baseline Klimatologi (2016-2025)</div>
        </div>
        <div class="kpi-card kpi-neutral">
            <div class="kpi-title">AKUMULASI BANJIR 10 TAHUN</div>
            <div class="kpi-value">{tot_flood} <span class="kpi-unit">kejadian</span></div>
            <div class="kpi-sub">{tot_houses:,} rumah terendam (2016-2025)</div>
        </div>
    </div>
    """


def render_kpi_cards(region, year, trend_data):
    row_now = trend_data[trend_data["year"] == int(year)]
    if row_now.empty:
        return render_projection_kpi_cards(year, trend_data)
    r = row_now.iloc[0]
    ir = float(r["dengue_incidence_rate"])
    cfr = float(r["dengue_case_fatality_rate"])
    rain = float(r["annual_rainfall_mm"])
    flood = int(r["flood_events_count"])
    houses = int(r["flooded_houses_count"])
    ir_alert = "kpi-alert" if ir > IR_THRESHOLD else "kpi-normal"
    ir_note = f"Melebihi Ambang ({IR_THRESHOLD})" if ir > IR_THRESHOLD else f"Di Bawah Ambang ({IR_THRESHOLD})"
    cfr_alert = "kpi-alert" if cfr > CFR_THRESHOLD else "kpi-normal"
    cfr_note = f"Melebihi Target (< {CFR_THRESHOLD}%)" if cfr > CFR_THRESHOLD else f"Sesuai Target (< {CFR_THRESHOLD}%)"
    return f"""
    <div class="kpi-grid">
        <div class="kpi-card {ir_alert}">
            <div class="kpi-title">INCIDENCE RATE (IR)</div>
            <div class="kpi-value">{ir:.2f} <span class="kpi-unit">/ 100rb</span></div>
            <div class="kpi-sub">{ir_note}</div>
        </div>
        <div class="kpi-card {cfr_alert}">
            <div class="kpi-title">CASE FATALITY RATE (CFR)</div>
            <div class="kpi-value">{cfr:.2f}%</div>
            <div class="kpi-sub">{cfr_note}</div>
        </div>
        <div class="kpi-card kpi-neutral">
            <div class="kpi-title">CURAH HUJAN TAHUNAN</div>
            <div class="kpi-value">{rain:,.1f} <span class="kpi-unit">mm</span></div>
            <div class="kpi-sub">Akumulasi Tahunan ({year})</div>
        </div>
        <div class="kpi-card kpi-neutral">
            <div class="kpi-title">DAMPAK BENCANA BANJIR</div>
            <div class="kpi-value">{flood} <span class="kpi-unit">kejadian</span></div>
            <div class="kpi-sub">{houses:,} rumah terendam</div>
        </div>
    </div>
    """


def render_stepper(plan):
    current = plan["status"] if plan else None
    steps = ["PENDING", "REJECTED"] if current == "REJECTED" else ["PENDING", "APPROVED", "DISPATCHED", "EXECUTED"]
    items_html = []
    for idx, s in enumerate(steps, start=1):
        cls = "step-item step-active" if s == current else "step-item"
        items_html.append(f"<div class='{cls}'><span class='step-num'>{idx}.</span> {s}</div>")
    return "<div class='stepper-bar'>" + "<div class='step-divider'></div>".join(items_html) + "</div>"


def prepare_map_dataframe(year):
    df = load_master_dataset()
    eff_year = 2025 if int(year) > 2025 else int(year)
    d = df[df["year"] == eff_year].copy()
    d["lat"] = d["regency_city"].map(lambda x: WEST_JAVA_COORDS.get(x, (-6.9, 107.6))[0])
    d["lon"] = d["regency_city"].map(lambda x: WEST_JAVA_COORDS.get(x, (-6.9, 107.6))[1])
    d["hover_info"] = (
        "<b>" + d["regency_city"] + f" ({eff_year})</b><br>"
        + "IR: " + d["dengue_incidence_rate"].round(2).astype(str) + " / 100rb<br>"
        + "CFR: " + d["dengue_case_fatality_rate"].round(2).astype(str) + "%<br>"
        + "Curah Hujan: " + d["annual_rainfall_mm"].round(1).astype(str) + " mm<br>"
        + "Banjir: " + d["flood_events_count"].astype(str) + " kejadian ("
        + d["flooded_houses_count"].astype(str) + " rumah)"
    )
    return d, eff_year


def build_region_focus_map(region, year):
    d, eff_year = prepare_map_dataframe(year)
    others = d[d["regency_city"].str.lower() != region.lower()]
    target = d[d["regency_city"].str.lower() == region.lower()]
    t_lat, t_lon = WEST_JAVA_COORDS.get(region, (-6.917, 107.619))
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=others["lat"], lon=others["lon"], mode="markers",
        name="Wilayah Sekitar (Jawa Barat)", text=others["hover_info"], hoverinfo="text",
        marker=dict(size=10, color="#64748B", opacity=0.75),
    ))
    fig.add_trace(go.Scattermap(
        lat=target["lat"], lon=target["lon"], mode="markers+text",
        name=f"Sasaran: {region}", text=[f"  {region}"], textposition="top right",
        hovertext=target["hover_info"], hoverinfo="text",
        marker=dict(size=22, color="#B91C1C", opacity=0.95),
    ))
    label_yr = f"Baseline 2025 untuk Proyeksi {year}" if int(year) > 2025 else str(eff_year)
    fig.update_layout(
        title=f"Peta Geospasial Fokus Wilayah: {region} ({label_yr})",
        map=dict(style="open-street-map", center=dict(lat=t_lat, lon=t_lon), zoom=9.2),
        paper_bgcolor="#FFFFFF", font=dict(color="#0F172A"),
        height=310, margin=dict(l=10, r=10, t=42, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig


def build_trend_figure(region, trend_data):
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_data["year"], y=trend_data["dengue_incidence_rate"],
        mode="lines+markers", name="Incidence Rate (/100rb)",
        line=dict(color="#B91C1C", width=3), marker=dict(size=8, color="#B91C1C"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_data["year"], y=trend_data["dengue_case_fatality_rate"],
        mode="lines+markers", name="Case Fatality Rate (%)",
        line=dict(color="#334155", width=2.5, dash="dot"), marker=dict(size=7, color="#334155"),
    ))
    fig_trend.add_hline(
        y=IR_THRESHOLD, line_dash="dash", line_color="#DC2626",
        annotation_text=f"Ambang IR Kemenkes ({IR_THRESHOLD})", annotation_position="top left",
    )
    fig_trend.update_layout(
        title=f"Tren Epidemiologi 10 Tahun: {region} (2016-2025)",
        xaxis_title="Tahun Pengamatan", yaxis_title="Nilai Indikator",
        template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"), height=310, margin=dict(l=30, r=20, t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_trend.update_xaxes(dtick=1, gridcolor="#E2E8F0")
    fig_trend.update_yaxes(gridcolor="#E2E8F0")
    return fig_trend


def build_provincial_map(d, year, highlight_region):
    high_risk = d[d["Kategori Risiko"].str.startswith("Risiko Tinggi")]
    controlled = d[~d["Kategori Risiko"].str.startswith("Risiko Tinggi")]
    center_lat, center_lon, zoom_lvl = -6.92, 107.60, 7.3
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermap(
        lat=controlled["lat"], lon=controlled["lon"], mode="markers",
        name="Terkendali (IR <= 49 & CFR <= 1.0%)", text=controlled["hover_info"], hoverinfo="text",
        marker=dict(size=12, color="#0F766E", opacity=0.85),
    ))
    fig_map.add_trace(go.Scattermap(
        lat=high_risk["lat"], lon=high_risk["lon"], mode="markers",
        name="Risiko Tinggi (IR > 49 atau CFR > 1.0%)", text=high_risk["hover_info"], hoverinfo="text",
        marker=dict(size=15, color="#B91C1C", opacity=0.9),
    ))
    if highlight_region and highlight_region != "Semua Wilayah (27 Kab/Kota)":
        sel = d[d["regency_city"] == highlight_region]
        center_lat, center_lon = WEST_JAVA_COORDS.get(highlight_region, (-6.92, 107.60))
        zoom_lvl = 9.0
        fig_map.add_trace(go.Scattermap(
            lat=sel["lat"], lon=sel["lon"], mode="markers+text",
            name=f"Sorotan: {highlight_region}", text=[f"  {highlight_region}"],
            textposition="top right", hovertext=sel["hover_info"], hoverinfo="text",
            marker=dict(size=24, color="#1E293B", opacity=1.0),
        ))
    fig_map.update_layout(
        title=f"Peta Geospasial Sebaran Risiko DBD 27 Kabupaten/Kota Jawa Barat ({year})",
        map=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=zoom_lvl),
        paper_bgcolor="#FFFFFF", font=dict(color="#0F172A"),
        height=390, margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig_map


def build_provincial_quadrant(d, year, highlight_region):
    d_plot = d.copy()
    d_plot["Ukuran_Banjir"] = d_plot["flooded_houses_count"].clip(lower=50)
    fig_quad = px.scatter(
        d_plot, x="annual_rainfall_mm", y="dengue_incidence_rate",
        size="Ukuran_Banjir", color="Kategori Risiko", hover_name="regency_city",
        hover_data=["dengue_case_fatality_rate", "Keterangan Ambang", "flood_events_count", "flooded_houses_count"],
        color_discrete_map={
            "Risiko Tinggi (IR > 49 atau CFR > 1.0%)": "#B91C1C",
            "Terkendali (IR <= 49 & CFR <= 1.0%)": "#0F766E",
        },
        title=f"Matriks Kuadran Multi-Risiko: Curah Hujan & Banjir vs Incidence Rate ({year})",
        labels={
            "annual_rainfall_mm": "Curah Hujan Tahunan (mm)",
            "dengue_incidence_rate": "Incidence Rate (/100rb Penduduk)",
            "dengue_case_fatality_rate": "CFR (%)",
        },
        template="plotly_white", size_max=34,
    )
    fig_quad.add_hline(y=IR_THRESHOLD, line_dash="dash", line_color="#7F1D1D", annotation_text=f"Ambang IR ({IR_THRESHOLD})")
    fig_quad.add_vline(x=float(d["annual_rainfall_mm"].median()), line_dash="dot", line_color="#64748B", annotation_text="Median Curah Hujan")
    if highlight_region and highlight_region != "Semua Wilayah (27 Kab/Kota)":
        sel = d[d["regency_city"] == highlight_region]
        if not sel.empty:
            r_row = sel.iloc[0]
            fig_quad.add_annotation(
                x=r_row["annual_rainfall_mm"], y=r_row["dengue_incidence_rate"],
                text=f"<b>{highlight_region}</b>", showarrow=True, arrowhead=2,
                arrowcolor="#0F172A", bgcolor="#FEF3C7", bordercolor="#B91C1C",
            )
    fig_quad.update_layout(
        height=390, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font=dict(color="#0F172A"),
        margin=dict(l=30, r=20, t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
    )
    return fig_quad


def generate_provincial_overview(year, highlight_region="Semua Wilayah (27 Kab/Kota)"):
    d, eff_year = prepare_map_dataframe(year)
    ir_exceed = d["dengue_incidence_rate"] > IR_THRESHOLD
    cfr_exceed = d["dengue_case_fatality_rate"] > CFR_THRESHOLD
    d["Kategori Risiko"] = (ir_exceed | cfr_exceed).map({
        True: "Risiko Tinggi (IR > 49 atau CFR > 1.0%)",
        False: "Terkendali (IR <= 49 & CFR <= 1.0%)",
    })
    d["Keterangan Ambang"] = (ir_exceed.astype(int) * 2 + cfr_exceed.astype(int)).map({
        3: "IR & CFR Melebihi Ambang",
        2: "IR Melebihi Ambang (> 49.0)",
        1: "Fatalitas CFR Melebihi Ambang (> 1.0%)",
        0: "IR & CFR Di Bawah Ambang",
    })
    fig_map = build_provincial_map(d, eff_year, highlight_region)
    fig_quad = build_provincial_quadrant(d, eff_year, highlight_region)
    d_sorted = d.sort_values("dengue_incidence_rate", ascending=False)
    fig_bar = px.bar(
        d_sorted, x="regency_city", y="dengue_incidence_rate", color="Kategori Risiko",
        hover_data=["dengue_case_fatality_rate", "Keterangan Ambang"],
        color_discrete_map={
            "Risiko Tinggi (IR > 49 atau CFR > 1.0%)": "#B91C1C",
            "Terkendali (IR <= 49 & CFR <= 1.0%)": "#94A3B8",
        },
        title=f"Peringkat Distribusi Incidence Rate (IR) 27 Kabupaten/Kota di Jawa Barat — Tahun {eff_year}",
        labels={
            "regency_city": "Kabupaten / Kota",
            "dengue_incidence_rate": "Incidence Rate (/100rb Penduduk)",
            "dengue_case_fatality_rate": "CFR (%)",
        },
        template="plotly_white",
    )
    fig_bar.add_hline(
        y=IR_THRESHOLD, line_dash="dash", line_color="#7F1D1D",
        annotation_text=f"Batas Ambang IR Kemenkes ({IR_THRESHOLD}/100rb) — *Merah jika IR > {IR_THRESHOLD} atau CFR > {CFR_THRESHOLD}%",
    )
    fig_bar.update_layout(
        xaxis_tickangle=-45, height=420, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color="#0F172A"), margin=dict(l=30, r=20, t=50, b=100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    table_view = d_sorted[[
        "regency_city", "dengue_incidence_rate", "dengue_case_fatality_rate",
        "annual_rainfall_mm", "flood_events_count", "flooded_houses_count",
        "Kategori Risiko", "Keterangan Ambang",
    ]].copy()
    table_view.columns = [
        "Kabupaten / Kota", "IR (/100rb)", "CFR (%)",
        "Curah Hujan (mm)", "Kejadian Banjir", "Rumah Terendam",
        "Status Evaluasi", "Pemicu Status",
    ]
    return fig_map, fig_quad, fig_bar, table_view.round(2)


# ---------------------------------------------------------------------
# 6. AI GATEWAY (QUERY BUILDER, TRACE, NATURAL Q&A, FAILOVER & FALLBACK)
# ---------------------------------------------------------------------
def build_rag_query(cfr_val, focus_mode, custom_notes):
    base_q = "Kriteria KLB fogging fokus radius 200 meter larvasidasi penyelidikan epidemiologi radius 100 meter pelaporan puskesmas"
    base_q += FOCUS_RAG_KEYWORDS.get(focus_mode, "")
    if cfr_val > CFR_THRESHOLD:
        base_q += " audit kematian tatalaksana syok dengue"
    if custom_notes.strip():
        base_q += " " + custom_notes.strip()
    return base_q


def format_sop_item(item):
    return f"**{item['source']}**\n> {item['content'].strip()[:260]}..."


def build_io_trace_markdown(region, year, focus_mode, tools_ms, epi_dict, clim_dict, hist_dict, sop_dict, rag_query):
    epi_out = {k: v for k, v in epi_dict.items() if k != "input_contract"}
    clim_out = {k: v for k, v in clim_dict.items() if k != "input_contract"}
    hist_summary = hist_dict.get("summary_10y", {})
    sop_excerpts_md = "\n\n".join(map(format_sop_item, sop_dict.get("results", [])))
    return (
        f"**Transparansi Kontrak Input-Output (Selesai dalam {tools_ms} ms):**\n"
        f"- **Mode Fokus Output Aktif:** `{focus_mode}`\n\n"
        f"**1. `tool_get_epidemiology`**\n"
        f"- `IN  (Argumen)` : `region='{region}', year={year}`\n"
        f"- `OUT (JSON)`    : `{json.dumps(epi_out, ensure_ascii=False)}`\n\n"
        f"**2. `tool_get_climate_flood`**\n"
        f"- `IN  (Argumen)` : `region='{region}', year={year}`\n"
        f"- `OUT (JSON)`    : `{json.dumps(clim_out, ensure_ascii=False)}`\n\n"
        f"**3. `tool_get_historical_trend`**\n"
        f"- `IN  (Argumen)` : `region='{region}' (Rentang 2016-2025, {hist_dict.get('records_count', 10)} baris)`\n"
        f"- `OUT (Ringkasan 10 Thn)` : `{json.dumps(hist_summary, ensure_ascii=False)}`\n\n"
        f"**4. `tool_query_kemenkes_sop`**\n"
        f"- `IN  (Argumen)` : `query='{rag_query[:65]}...', n_results=3`\n"
        f"- `OUT (Kutipan Regulasi)` :\n\n{sop_excerpts_md}"
    )


def rule_based_fallback(region, year, focus_mode, epi_dict, clim_dict, hist_dict, custom_notes=""):
    ir = epi_dict.get("incidence_rate", epi_dict.get("baseline_mean_ir_10y", 0.0))
    cfr = epi_dict.get("case_fatality_rate", epi_dict.get("baseline_mean_cfr_10y", 0.0))
    summary_10y = hist_dict.get("summary_10y", {})
    rain = clim_dict.get("annual_rainfall_mm", clim_dict.get("avg_annual_rainfall_10y_mm", 0.0))
    flood = clim_dict.get("flood_events_count", clim_dict.get("avg_flood_events_10y", 0))
    houses = clim_dict.get("flooded_houses_count", clim_dict.get("total_flooded_houses_10y", 0))
    status_level = "TINGGI (DARURAT INTERVENSI)" if (ir > IR_THRESHOLD or cfr > CFR_THRESHOLD) else "SEDANG - TERKENDALI"
    notes_line = f"- **Catatan/Pertanyaan Lapangan Petugas:** {custom_notes.strip()}\n" if custom_notes.strip() else ""
    return (
        f"### LAPORAN CADANGAN DETERMINISTIK (RULE-BASED FALLBACK)\n"
        f"*Koneksi API Gemini belum tersedia atau mencapai batas kuota. Laporan disusun otomatis oleh mesin aturan deterministik.*\n\n"
        f"**Wilayah Evaluasi:** {region} ({year})  \n"
        f"**Mode Fokus:** {focus_mode}  \n"
        f"**Klasifikasi Risiko:** {status_level}\n{notes_line}\n"
        f"#### 1. Evaluasi Indikator Epidemiologi & Baseline 10 Tahun (2016-2025)\n"
        f"- Incidence Rate Evaluasi / Baseline: **{ir:.2f} per 100.000 penduduk** (Ambang Nasional: {IR_THRESHOLD})\n"
        f"- Case Fatality Rate Evaluasi / Baseline: **{cfr:.2f}%** (Target Nasional: < {CFR_THRESHOLD}%)\n"
        f"- Rata-rata IR 10 Tahun: **{summary_10y.get('mean_ir_10y', 0):.2f}** | Tahun Puncak Siklus: **{summary_10y.get('peak_outbreak_year', '-')}** (IR: {summary_10y.get('peak_outbreak_ir', 0):.2f})\n\n"
        f"#### 2. Faktor Risiko Iklim dan Bencana Banjir DIBI BNPB\n"
        f"- Curah Hujan: **{rain:,.1f} mm**\n"
        f"- Kejadian Banjir: **{flood} kejadian** ({houses:,} rumah terendam)\n\n"
        f"#### 3. Protokol Standar Wajib (Pedoman Kemenkes RI 2017) & Proyeksi (2026+)\n"
        f"1. **Penyelidikan Epidemiologi (PE):** Dilaksanakan pada radius minimal 100 meter dari lokasi penderita (Halaman 94).\n"
        f"2. **Pengendalian Vektor Fokus:** PSN 3M Plus, larvasidasi selektif, dan fogging fokus 2 siklus interval 7 hari pada radius 200 meter apabila memenuhi kriteria teknis (Halaman 99).\n"
        f"3. **Pelaporan Berjenjang & Kewaspadaan Dini (2026+):** Laporan disampaikan Puskesmas ke Dinas Kesehatan Kabupaten/Kota dengan tembusan Camat dan Lurah (Halaman 98), disertai pemantauan mingguan SKD-KLB pra-musim hujan (Halaman 102-103).\n\n"
        f"STATUS: PENDING - Menunggu persetujuan resmi Kepala Daerah."
    )


def invoke_gemini_with_failover(prompt, logger):
    gem_client = get_gemini_client()
    if gem_client is None:
        return "", "", RuntimeError("GEMINI_API_KEY belum dikonfigurasi pada environment/Secrets.")
    last_err = None
    for candidate_model in MODEL_CANDIDATES:
        try:
            t0 = time.time()
            resp = gem_client.models.generate_content(
                model=candidate_model, contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            dur = round(time.time() - t0, 2)
            source_used = f"Agentic AI ({candidate_model} - {dur}s)"
            logger.log("DengueGuard-Agent", "GENERATE_PLAN", "-", f"Model: {candidate_model}, Durasi: {dur}s")
            return resp.text, source_used, None
        except Exception as e:
            last_err = e
    return "", "", last_err


def execute_agentic_analysis(region, year_selection, focus_mode, officer_name, custom_notes):
    logger = get_audit_logger()
    year = parse_year_selection(year_selection)
    effective_focus = FOCUS_MODES[4] if year > 2025 else focus_mode
    actor = officer_name.strip() or "Petugas Dinkes"
    logger.log(actor, "REQUEST_ANALYSIS", "-", f"Target: {region} ({year}) | {effective_focus[:28]}")

    t_start = time.time()
    epi_json = tool_get_epidemiology(region, year)
    epi_dict = json.loads(epi_json)
    clim_json = tool_get_climate_flood(region, year)
    clim_dict = json.loads(clim_json)
    hist_json = tool_get_historical_trend(region)
    hist_dict = json.loads(hist_json)

    cfr_val = epi_dict.get("case_fatality_rate", epi_dict.get("baseline_mean_cfr_10y", 0.0))
    rag_query = build_rag_query(cfr_val, effective_focus, custom_notes)
    sop_json = tool_query_kemenkes_sop(rag_query, n_results=3)
    sop_dict = json.loads(sop_json)
    tools_ms = round((time.time() - t_start) * 1000, 1)

    trace_md = build_io_trace_markdown(
        region, year, effective_focus, tools_ms, epi_dict, clim_dict, hist_dict, sop_dict, rag_query
    )
    focus_directive = FOCUS_INSTRUCTIONS.get(effective_focus, FOCUS_INSTRUCTIONS[FOCUS_MODES[0]])
    agent_prompt = f"""Anda adalah 'DengueGuard-Agent', sistem pakar pendukung keputusan pengendalian DBD Provinsi Jawa Barat.

Kontrak Input-Output 4 alat bantu (MCP Tools) untuk {region} tahun evaluasi {year}:
[1. tool_get_epidemiology]: {epi_json}
[2. tool_get_historical_trend (2016-2025)]: {hist_json}
[3. tool_get_climate_flood (DIBI BNPB & Open-Meteo)]: {clim_json}
[4. tool_query_kemenkes_sop (Kemenkes RI 2017)]: {sop_json}

MODE FOKUS OUTPUT PILIHAN PENGGUNA: {effective_focus}
PERTANYAAN NATURAL / CATATAN LAPANGAN PETUGAS: {custom_notes.strip() or 'Tidak ada catatan khusus.'}

BATASAN DOMAIN & ATURAN KETAT:
1. Jawab dengan bahasa analisis kesehatan masyarakat yang natural dan jelas, namun TOLAK secara sopan apabila pertanyaan tambahan tidak berkaitan dengan DBD, epidemiologi, iklim/banjir, vektor Aedes, atau SOP Kemenkes 2017.
2. Gunakan HANYA fakta angka dari payload JSON di atas (ambang IR nasional = 49.0/100rb, target CFR nasional < 1.0%).
3. Apabila terdapat PERTANYAAN NATURAL / CATATAN LAPANGAN PETUGAS yang relevan, jawab pertanyaan tersebut secara langsung pada bagian awal atau sub-bagian rekomendasi tindakan berdasarkan data wilayah dan rujukan Kemenkes 2017.
4. Jika status tool adalah 'out_of_dataset_projection' (Tahun {year} di luar 2016-2025), nyatakan secara eksplisit bahwa tahun {year} berada di luar dataset historis dan gunakan baseline 10 tahun (2016-2025) tanpa mengarang angka baru.
5. Wajib kutip nomor halaman Pedoman Kemenkes RI 2017 (Halaman 94 untuk PE radius 100m, Halaman 98 pelaporan Puskesmas ke Camat/Lurah, Halaman 99 penanggulangan fokus radius 200m & fogging 2 siklus, Halaman 102-103 SKD-KLB).

{focus_directive}

Akhiri laporan dengan kalimat persis:
STATUS: PENDING - Menunggu persetujuan resmi Kepala Daerah."""

    report_text, source_used, last_err = invoke_gemini_with_failover(agent_prompt, logger)
    if last_err is not None or not report_text:
        report_text = rule_based_fallback(region, year, effective_focus, epi_dict, clim_dict, hist_dict, custom_notes)
        source_used = "Rule-Based Fallback"
        logger.log("AI-Gateway", "FALLBACK_TRIGGERED", "-", f"Error: {str(last_err)[:80]}")

    plan = new_plan(region, year, report_text, source=source_used, focus_mode=effective_focus)
    logger.log("DengueGuard-Agent", "CREATE_PLAN", plan["plan_id"], f"Status: PENDING ({source_used})")
    return plan, trace_md, report_text


# ---------------------------------------------------------------------
# 7. TEMA CSS & PENATAAN HALAMAN STREAMLIT (LANDING, DINKES, BUPATI)
# ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #F8FAFC !important;
    color: #0F172A !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1260px !important;
}
.header-box {
    background: #FFFFFF; border: 1px solid #CBD5E1; border-top: 5px solid #B91C1C;
    padding: 16px 24px; border-radius: 8px; margin-bottom: 14px;
    display: flex; justify-content: space-between; align-items: center; gap: 20px;
}
.header-text-col { flex: 1; }
.header-logo-wrap {
    width: 68px; height: 68px; border-radius: 12px; background: #FFF5F5;
    border: 1.5px solid #FECACA; display: flex; align-items: center; justify-content: center; padding: 6px;
}
.header-logo-img { width: 100%; height: 100%; object-fit: contain; }
.header-sup { font-size: 11px; font-weight: 700; letter-spacing: 1.2px; color: #B91C1C; text-transform: uppercase; margin-bottom: 4px; }
.header-title { font-size: 21px; font-weight: 700; color: #0F172A; margin: 0 0 4px 0; }
.header-sub { font-size: 13.5px; color: #475569; margin: 0; }
.intro-box {
    background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 5px solid #0F766E;
    border-radius: 8px; padding: 18px 22px; margin-bottom: 14px;
}
.intro-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 22px; }
@media (max-width: 900px) { .intro-grid { grid-template-columns: 1fr; } }
.intro-kicker { font-size: 11px; font-weight: 700; letter-spacing: 0.9px; color: #B91C1C; margin-bottom: 6px; }
.intro-heading { font-size: 17px; font-weight: 700; margin: 0 0 8px 0; color: #0F172A; }
.intro-p { font-size: 13.5px; line-height: 1.55; margin: 0 0 10px 0; color: #334155; }
.workflow-steps { display: flex; flex-direction: column; gap: 7px; }
.wf-step { background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 6px; padding: 8px 12px; font-size: 12.8px; line-height: 1.45; color: #0F172A; }
.glossary-list { margin: 0; display: grid; grid-template-columns: 1fr; gap: 6px; }
.glossary-list dt { font-size: 12.5px; font-weight: 700; color: #0F172A; margin-top: 2px; }
.glossary-list dd { font-size: 12.3px; color: #334155; margin: 0 0 4px 0; line-height: 1.4; border-bottom: 1px dashed #E2E8F0; padding-bottom: 4px; }
.role-card-box {
    background: #FFFFFF; border: 1.5px solid #CBD5E1; border-top: 4px solid #B91C1C;
    border-radius: 8px; padding: 16px 20px; margin-bottom: 10px;
}
.role-card-bupati { border-top-color: #1E293B; }
.role-bar {
    background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 5px solid #0F766E;
    border-radius: 6px; padding: 10px 16px; font-size: 13.5px; color: #0F172A; margin-bottom: 8px;
}
.status-banner {
    background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 5px solid #B91C1C;
    padding: 12px 20px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;
}
.status-label { font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 0.8px; }
.status-main { font-size: 14px; font-weight: 600; color: #0F172A; margin-top: 2px; }
.status-meta { font-size: 13px; color: #475569; }
.status-pill { padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 700; }
.badge-pending { background: #FEF3C7; color: #92400E; border: 1px solid #F59E0B; }
.badge-approved { background: #DCFCE7; color: #166534; border: 1px solid #22C55E; }
.badge-dispatched { background: #DBEAFE; color: #1E40AF; border: 1px solid #3B82F6; }
.badge-executed { background: #F1F5F9; color: #0F172A; border: 1px solid #64748B; }
.badge-rejected { background: #FEE2E2; color: #991B1B; border: 1px solid #EF4444; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 8px 0 16px 0; }
@media (max-width: 900px) { .kpi-grid { grid-template-columns: repeat(2, 1fr); } }
.kpi-card { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 14px 16px; }
.kpi-alert { border-top: 4px solid #B91C1C; background: #FFFDFD; }
.kpi-normal { border-top: 4px solid #0F766E; }
.kpi-neutral { border-top: 4px solid #334155; }
.kpi-title { font-size: 11px; font-weight: 700; color: #475569; letter-spacing: 0.6px; }
.kpi-value { font-size: 22px; font-weight: 700; color: #0F172A; margin: 4px 0; }
.kpi-unit { font-size: 12px; font-weight: 500; color: #475569; }
.kpi-sub { font-size: 12px; color: #334155; }
.stepper-bar { display: flex; align-items: center; background: #FFFFFF; border: 1px solid #CBD5E1; padding: 12px 18px; border-radius: 8px; margin-bottom: 14px; gap: 10px; }
.step-item { font-size: 12px; font-weight: 600; color: #334155; padding: 6px 14px; border-radius: 6px; background: #F1F5F9; }
.step-active { background: #B91C1C; color: #FFFFFF; }
.step-divider { flex: 1; height: 1px; background: #CBD5E1; }
.msg-box { padding: 12px 16px; border-radius: 6px; font-size: 13.5px; margin: 8px 0; }
.msg-ok { background: #F0FDF4; border: 1px solid #86EFAC; color: #166534; }
.msg-err { background: #FEF2F2; border: 1px solid #FCA5A5; color: #991B1B; }
.msg-warn { background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }
.footer-box {
    margin-top: 26px; padding: 14px 20px; background: #FFFFFF; border: 1px solid #CBD5E1;
    border-bottom: 4px solid #B91C1C; border-radius: 8px; text-align: center;
    font-size: 12.5px; color: #475569; line-height: 1.55;
}
.footer-rights { font-weight: 700; color: #0F172A; margin-top: 2px; letter-spacing: 0.4px; }
</style>
"""


def init_session_state():
    defaults = {
        "role": "landing",
        "active_plan": None,
        "last_region": "Kota Depok",
        "last_year": 2024,
        "last_trace": "*Kontrak Input dan Output JSON dari keempat MCP Tools serta kutipan SOP Kemenkes akan tampil di sini setelah analisis dijalankan.*",
        "last_report": "Silakan pilih wilayah, tahun, dan mode fokus di atas, lalu tekan tombol **Jalankan Analisis Agentic AI**.",
        "guardrail_html": "",
        "hitl_msg": "",
        "audit_verify_html": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_global_header():
    logo_html = load_logo_html()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="header-box">
        <div class="header-text-col">
            <div class="header-sup">Pemerintah Provinsi Jawa Barat &bull; Dinas Kesehatan</div>
            <div class="header-title">DengueGuard-Agent: Sistem Pendukung Keputusan Mitigasi DBD</div>
            <div class="header-sub">Integrasi Surveilans Epidemiologi 10 Tahun (2016–2025), Pemodelan Geospasial Iklim-Banjir DIBI BNPB, dan Regulasi Pedoman Kemenkes RI 2017</div>
        </div>
        {logo_html}
    </div>
    """, unsafe_allow_html=True)


def render_role_top_bar(role_title, access_desc):
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(
            f"<div class='role-bar'><div>Peran Aktif: <strong>{role_title}</strong> &nbsp;&bull;&nbsp; Hak Akses: {access_desc}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Kembali ke Beranda / Ganti Peran", key=f"back_{st.session_state.role}", width="stretch"):
            st.session_state.role = "landing"
            st.session_state.hitl_msg = ""
            st.rerun()
    st.markdown(render_status_bar(st.session_state.active_plan), unsafe_allow_html=True)
    with st.expander("Panduan Penggunaan & Glosarium Istilah Sistem (Klik untuk Membuka)", expanded=False):
        st.markdown(render_intro_glossary_html(), unsafe_allow_html=True)


def render_landing_page():
    st.markdown(render_intro_glossary_html(), unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="role-card-box">
            <h3 style="margin-top:0;">Portal 1: Analis Dinas Kesehatan (P2P)</h3>
            <p>Akses operasional untuk analis surveilans dan tim lapangan Dinkes / Puskesmas:</p>
            <ul>
                <li>Menjalankan investigasi agen berbasis 4 MCP Tools &amp; RAG Kemenkes RI 2017.</li>
                <li>Mengajukan pertanyaan natural atau catatan kondisi lapangan (dilindungi guardrail topik).</li>
                <li>Menyusun draf rencana intervensi (<code>PENDING</code>) dan mengeksekusi perintah lapangan (<em>Dispatch &amp; Execute</em>) setelah disetujui.</li>
                <li>Memantau Ringkasan Risiko 27 Kab/Kota Jawa Barat serta Audit Log SHA-256.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk sebagai Analis Dinas Kesehatan (P2P)", type="primary", key="enter_dinkes", width="stretch"):
            st.session_state.role = "dinkes"
            st.session_state.hitl_msg = ""
            st.rerun()

    with c2:
        st.markdown("""
        <div class="role-card-box role-card-bupati">
            <h3 style="margin-top:0;">Portal 2: Kepala Daerah (Bupati / Walikota)</h3>
            <p>Gerbang otorisasi eksekutif (<em>Human-in-the-Loop</em>) bagi pimpinan wilayah:</p>
            <ul>
                <li>Meninjau draf rencana mitigasi aktif berstatus <code>PENDING</code> yang diajukan Analis Dinkes.</li>
                <li>Mengambil keputusan sah <strong>Approve</strong> (Setujui) atau <strong>Reject</strong> (Tolak beserta alasan wajib).</li>
                <li>Memantau Peta Geospasial dan Kuadran Risiko Provinsi Jawa Barat sebelum menandatangani persetujuan.</li>
                <li>Mengaudit riwayat transaksi keputusan pada log kriptografis SHA-256.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Masuk sebagai Kepala Daerah (Bupati / Walikota)", type="primary", key="enter_bupati", width="stretch"):
            st.session_state.role = "bupati"
            st.session_state.hitl_msg = ""
            st.rerun()


def render_provincial_tab(prefix, historical_years, prov_region_options):
    col_yr, col_reg, col_desc = st.columns([1, 1, 2])
    with col_yr:
        yr_idx = historical_years.index("2024") if "2024" in historical_years else 0
        prov_yr = st.selectbox("Pilih Tahun Evaluasi Provinsi", historical_years, index=yr_idx, key=f"{prefix}_prov_yr")
    with col_reg:
        def_idx = prov_region_options.index("Kota Depok") if "Kota Depok" in prov_region_options else 0
        prov_reg = st.selectbox("Sorot Spesifik Kabupaten / Kota pada Peta & Grafik", prov_region_options, index=def_idx, key=f"{prefix}_prov_reg")
    with col_desc:
        st.info("Analisis Geospasial & Kuadran Multi-Risiko Jawa Barat (27 Kab/Kota): Sorot wilayah tertentu untuk zoom-in pada peta dan kuadran.")

    fig_map, fig_quad, fig_bar, tbl = generate_provincial_overview(prov_yr, prov_reg)
    m1, m2 = st.columns(2)
    with m1:
        st.plotly_chart(fig_map, width="stretch", key=f"{prefix}_prov_map")
    with m2:
        st.plotly_chart(fig_quad, width="stretch", key=f"{prefix}_prov_quad")
    st.plotly_chart(fig_bar, width="stretch", key=f"{prefix}_prov_bar")
    st.dataframe(tbl, width="stretch", hide_index=True, key=f"{prefix}_prov_tbl")


def render_audit_tab(prefix):
    st.markdown("#### Buku Besar Audit Anti-Manipulasi (*Tamper-Evident Audit Ledger*)")
    logger = get_audit_logger()
    if st.button("Verifikasi Integritas Rantai Hash SHA-256", type="primary", key=f"{prefix}_chk_audit"):
        st.session_state.audit_verify_html = logger.verify()
    if st.session_state.audit_verify_html:
        st.markdown(st.session_state.audit_verify_html, unsafe_allow_html=True)
    st.dataframe(logger.table(), width="stretch", hide_index=True, key=f"{prefix}_audit_tbl")


def render_dinkes_analysis_tab(regions, year_options):
    df = load_master_dataset()
    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 13px; color: #DC2626; font-weight: 600; margin-bottom: 6px;'>"
            "* Catatan: Kolom bertanda bintang (*) Wajib Diisi sebelum menjalankan analisis.</div>",
            unsafe_allow_html=True,
        )
        r1, r2, r3 = st.columns(3)
        with r1:
            sel_reg = st.selectbox("Kabupaten / Kota Sasaran (Jawa Barat) *", [""] + regions, index=0, key="dinkes_reg")
        with r2:
            sel_yr = st.selectbox("Tahun Evaluasi / Proyeksi *", [""] + year_options, index=0, key="dinkes_yr")
        with r3:
            officer_name = st.text_input("Nama Petugas Analis (Dinkes)", value="", key="dinkes_officer")

        focus_mode = st.radio(
            "Pilih Mode / Ruang Lingkup Output Spesifik (Mengatur Fokus Analisis & Tools)",
            FOCUS_MODES, index=0, key="dinkes_focus",
        )
        n1, n2 = st.columns([4, 1], vertical_alignment="bottom")
        with n1:
            custom_notes = st.text_input(
                "Pertanyaan Natural / Catatan Kondisi Lapangan (Opsional — Dilindungi Guardrail Topik DBD)",
                placeholder="Contoh: 'Minggu lalu banjir 3 hari di kelurahan padat, langkah paling mendesak apa sesuai Kemenkes?' atau kosongkan.",
                key="dinkes_notes",
            )
        with n2:
            run_clicked = st.button("Jalankan Analisis Agentic AI", type="primary", key="dinkes_run", width="stretch")

    if run_clicked:
        _process_dinkes_run(sel_reg, sel_yr, focus_mode, officer_name, custom_notes)

    if st.session_state.guardrail_html:
        st.markdown(st.session_state.guardrail_html, unsafe_allow_html=True)

    disp_reg = sel_reg if sel_reg else st.session_state.last_region
    disp_yr = parse_year_selection(sel_yr) if sel_yr else st.session_state.last_year
    trend_data = df[df["regency_city"].str.lower() == disp_reg.lower()].sort_values("year")
    st.markdown(render_kpi_cards(disp_reg, disp_yr, trend_data), unsafe_allow_html=True)

    c_left, c_right = st.columns([5, 7])
    with c_left:
        st.markdown("#### Visualisasi Tren Historis 10 Tahun (2016–2025)")
        st.plotly_chart(build_trend_figure(disp_reg, trend_data), width="stretch", key="dinkes_trend")
        st.markdown("#### Peta Geospasial Fokus Wilayah Sasaran (Jawa Barat)")
        st.plotly_chart(build_region_focus_map(disp_reg, disp_yr), width="stretch", key="dinkes_focus_map")
        with st.container(border=True):
            st.markdown("#### Kontrak Input-Output Alat Agen (I/O Trace) & Kutipan Regulasi")
            st.markdown(st.session_state.last_trace)
    with c_right:
        with st.container(border=True):
            st.markdown("#### Draf Laporan Investigasi & Rekomendasi Mitigasi")
            st.markdown(st.session_state.last_report)


def _process_dinkes_run(sel_reg, sel_yr, focus_mode, officer_name, custom_notes):
    if not sel_reg or not str(sel_yr).strip():
        st.session_state.guardrail_html = (
            "<div class='msg-box msg-err'><strong>Validasi Input:</strong> "
            "Kabupaten/Kota Sasaran dan Tahun Evaluasi wajib dipilih terlebih dahulu.</div>"
        )
        return
    logger = get_audit_logger()
    actor = officer_name.strip() or "Petugas Dinkes"
    allowed, guard_code, guard_msg = check_topic_guardrail(custom_notes)
    if not allowed:
        logger.log(actor, guard_code, "-", f"Input ditolak: {custom_notes.strip()[:45]}")
        st.session_state.guardrail_html = f"<div class='msg-box msg-err'><strong>Guardrail Topik Aktif ({guard_code}):</strong> {guard_msg}</div>"
        st.session_state.last_trace = f"**Guardrail Audit ({guard_code}):** Eksekusi agen dihentikan sebelum memanggil LLM."
        st.session_state.last_report = (
            f"### Permintaan Ditolak oleh Guardrail Keamanan Topik\n\n{guard_msg}\n\n"
            "Silakan ajukan pertanyaan atau catatan seputar **epidemiologi DBD, iklim/banjir Jawa Barat, vektor nyamuk, faskes, atau SOP Kemenkes RI 2017**."
        )
        return

    with st.spinner("Menjalankan 4 MCP Tools, kueri ChromaDB Kemenkes 2017, dan sintesis Gemini..."):
        plan, trace_md, report_text = execute_agentic_analysis(sel_reg, sel_yr, focus_mode, officer_name, custom_notes)
    st.session_state.guardrail_html = ""
    st.session_state.active_plan = plan
    st.session_state.last_region = sel_reg
    st.session_state.last_year = parse_year_selection(sel_yr)
    st.session_state.last_trace = trace_md
    st.session_state.last_report = report_text
    st.rerun()


def render_dinkes_dispatch_tab():
    st.markdown(
        "#### Eksekusi Lapangan Tim Dinas Kesehatan & Puskesmas\n"
        "*Catatan Tata Kelola: Tombol persetujuan (`Approve` / `Reject`) khusus berada pada Portal Kepala Daerah. "
        "Petugas Dinkes mengirim perintah lapangan (`Dispatch`) setelah draf berstatus **APPROVED**.*"
    )
    st.markdown(render_stepper(st.session_state.active_plan), unsafe_allow_html=True)
    c_left, c_right = st.columns([7, 5])
    with c_left:
        with st.container(border=True):
            st.markdown(render_plan_preview_md(st.session_state.active_plan))
    with c_right:
        with st.container(border=True):
            field_actor = st.text_input(
                "Nama Petugas Pelaksana Lapangan (Dinkes / Puskesmas)",
                placeholder="Contoh: dr. Farhan / Tim P2P Puskesmas",
                key="dinkes_field_actor",
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("1. Kirim Instruksi (Dispatch)", type="primary", key="btn_disp", width="stretch"):
                    _run_hitl_transition("DISPATCH", field_actor)
            with b2:
                if st.button("2. Tandai Selesai (Execute)", type="secondary", key="btn_exec", width="stretch"):
                    _run_hitl_transition("EXECUTE", field_actor)
        if st.session_state.hitl_msg:
            st.markdown(st.session_state.hitl_msg, unsafe_allow_html=True)


def render_bupati_auth_tab():
    st.markdown(
        "#### Gerbang Keputusan Resmi Kepala Daerah (*Executive State Guard*)\n"
        "Tinjau draf rencana mitigasi yang diajukan Analis Dinas Kesehatan sebelum memberikan pengesahan atau penolakan."
    )
    st.markdown(render_stepper(st.session_state.active_plan), unsafe_allow_html=True)
    c_left, c_right = st.columns([7, 5])
    with c_left:
        with st.container(border=True):
            st.markdown(render_plan_preview_md(st.session_state.active_plan))
    with c_right:
        with st.container(border=True):
            auth_name = st.text_input(
                "Nama Pejabat Otorisasi (Bupati / Walikota)",
                placeholder="Contoh: Wali Kota Depok / Bupati Tasikmalaya",
                key="bupati_auth_name",
            )
            st.markdown(f"<div class='msg-box msg-ok'>Peran Terverifikasi: <strong>{APPROVER_ROLE}</strong></div>", unsafe_allow_html=True)
            reject_reason = st.text_area(
                "Catatan / Alasan Penolakan (Wajib diisi jika memilih Tolak Draf)",
                placeholder="Tuliskan arahan koreksi atau alasan penolakan rencana...",
                height=100, key="bupati_reject_reason",
            )
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Setujui Draf (Approve)", type="primary", key="btn_appr", width="stretch"):
                    _run_hitl_transition("APPROVE", auth_name)
            with b2:
                if st.button("Tolak Draf (Reject)", type="secondary", key="btn_rejc", width="stretch"):
                    _run_hitl_transition("REJECT", auth_name, reject_reason)
        if st.session_state.hitl_msg:
            st.markdown(st.session_state.hitl_msg, unsafe_allow_html=True)


def _run_hitl_transition(action_type, actor_name, reason=""):
    logger = get_audit_logger()
    plan = st.session_state.active_plan
    try:
        if action_type == "APPROVE":
            updated = approve_plan(plan, actor_name, APPROVER_ROLE, logger)
        elif action_type == "REJECT":
            updated = reject_plan(plan, actor_name, APPROVER_ROLE, reason, logger)
        elif action_type == "DISPATCH":
            updated = dispatch_plan(plan, actor_name, logger)
        else:
            updated = execute_plan(plan, actor_name, logger)
        st.session_state.active_plan = updated
        st.session_state.hitl_msg = (
            f"<div class='msg-box msg-ok'><strong>Transisi Status Berhasil:</strong> Rencana "
            f"<code>{updated['plan_id']}</code> kini berstatus <strong>{updated['status']}</strong>.</div>"
        )
        st.rerun()
    except ValueError as err:
        st.session_state.hitl_msg = f"<div class='msg-box msg-err'><strong>Penolakan Sistem (State Guard):</strong> {str(err)}</div>"


def main():
    st.set_page_config(
        page_title="DengueGuard-Agent | Sistem Pendukung Keputusan DBD Jawa Barat",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    init_session_state()
    render_global_header()

    df = load_master_dataset()
    regions = sorted(df["regency_city"].unique().tolist())
    prov_region_options = ["Semua Wilayah (27 Kab/Kota)"] + regions
    historical_years = [str(y) for y in sorted(int(y) for y in df["year"].unique())]
    year_options = historical_years + ["2026 (Proyeksi Tren 10 Tahun)"]

    role = st.session_state.role
    if role == "landing":
        render_landing_page()
    elif role == "dinkes":
        render_role_top_bar("Analis Dinas Kesehatan (P2P)", "Investigasi AI, Pengajuan Draf, Dispatch & Execute")
        t1, t2, t3, t4 = st.tabs([
            "1. Analisis Risiko & Investigasi Agen",
            "2. Tindak Lanjut Lapangan (Dispatch & Execute)",
            "3. Ringkasan Risiko Provinsi Jawa Barat",
            "4. Rekam Jejak Audit Kriptografis (SHA-256)",
        ])
        with t1:
            render_dinkes_analysis_tab(regions, year_options)
        with t2:
            render_dinkes_dispatch_tab()
        with t3:
            render_provincial_tab("dinkes", historical_years, prov_region_options)
        with t4:
            render_audit_tab("dinkes")
    else:
        render_role_top_bar("Kepala Daerah (Bupati / Walikota)", "Otorisasi Eksekutif (Approve / Reject)")
        t1, t2, t3 = st.tabs([
            "1. Otorisasi Eksekutif (Human-in-the-Loop)",
            "2. Ringkasan Risiko Provinsi Jawa Barat",
            "3. Rekam Jejak Audit Kriptografis (SHA-256)",
        ])
        with t1:
            render_bupati_auth_tab()
        with t2:
            render_provincial_tab("bupati", historical_years, prov_region_options)
        with t3:
            render_audit_tab("bupati")

    st.markdown(render_footer_html(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
