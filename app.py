"""
Comprehensive Lab Extractor & CloudTrail Analyzer — Educative CloudLabs + S3 + Gemini AI

This merged application provides:
1. Lab content extraction from Educative CloudLabs (Playwright + Requests)
2. S3 storage for content, policies, prescripts, and execution logs
3. CloudTrail log processing and analysis
4. Gemini AI-powered lab issue analysis and resource compliance checking
5. Resource configuration extraction using OpenAI

Features:
- Full lab content extraction with ACE editor support
- Execution logs fetching and storage
- CloudTrail log processing with filtering and pagination
- AI-powered analysis of lab issues and resource compliance
- S3-based file management and viewing
"""



from __future__ import annotations

import io
import re
import os
import json
import time
import pathlib
import hashlib
import subprocess
import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict

# --- NLTK bootstrap (must be before any llama_index imports) ---
import os
import nltk

# choose a writable dir on Streamlit Cloud
NLTK_DIR = os.path.join(os.path.expanduser("~"), ".cache", "nltk")
os.makedirs(NLTK_DIR, exist_ok=True)

# Ensure NLTK searches this path first
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

# Make sure required packages exist; download only if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=NLTK_DIR)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DIR)
# --- end NLTK bootstrap ---

import sys
try:
    __import__("pysqlite3")                 # loads the wheel with modern SQLite
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    # If the wheel isn't present, Chroma will try its own fallback and raise
    # the same error; this keeps local dev working if you already have new sqlite.
    pass


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
import google.generativeai as genai
import traceback

import requests
from bs4 import BeautifulSoup, Tag
from urllib.parse import urlparse, urljoin, quote_plus

import streamlit as st

@st.cache_resource(show_spinner=False)
def ensure_chromium_installed():
    """Download Playwright Chromium browser if missing (idempotent)."""
    # Put browsers in a writable, cached location
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", os.path.join(os.path.expanduser("~"), ".cache", "ms-playwright"))
    # This command is safe to run repeatedly; it will no-op if already installed
    subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
# --- Google Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# --- Async HTTP
try:
    import aiohttp
except Exception:
    aiohttp = None

# --- S3 imports
try:
    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError
except Exception:
    boto3 = None

# --- OpenAI
from llama_index.core import Settings
# from llama_index.embeddings.openai import OpenAIEmbedding

st.set_page_config(
    layout="wide",
    page_title="Lab Extractor & CloudTrail Analyzer",
    initial_sidebar_state="collapsed",
)

OPENAI_KEY = (st.secrets.get("openaikey") or {}).get("key") or os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_KEY)






# -----------------------------
# Constants & data classes
# -----------------------------
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/138.0.0.0 Safari/537.36"
)

@dataclass
class TaskLink:
    category: Optional[str]
    title: str
    href: str  # absolute URL

@dataclass
class TaskArtifact:
    url: str
    title: str
    slate_text: str
    code_blocks: List[Dict[str, str]]  # [{"type": "ace", "content": "..."}]

# -----------------------------
# Utilities
# -----------------------------
def normalize_url(base: str, link: str) -> Optional[str]:
    try:
        url = urljoin(base, link)
        parsed = urlparse(url)
        if parsed.scheme in {"http", "https"}:
            return parsed._replace(fragment="").geturl()
    except Exception:
        return None
    return None


def build_session(raw_headers: str, timeout: int = 30) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    # Parse simple key:value headers pasted by the user
    for line in (raw_headers or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            s.headers[k.strip()] = v.strip()
        else:
            if line.lower().startswith("cookie="):
                s.headers["Cookie"] = line.split("=", 1)[1].strip()
            elif line.lower().startswith("authorization="):
                s.headers["Authorization"] = line.split("=", 1)[1].strip()
    return s


def parse_cookie_header_to_playwright_cookies(cookie_header: str, url: str) -> List[dict]:
    cookies: List[dict] = []
    if not cookie_header:
        return cookies
    host = urlparse(url).hostname or ""
    domain = host[4:] if host.startswith("www.") else host
    for part in cookie_header.split(';'):
        if '=' not in part:
            continue
        name, value = part.split('=', 1)
        name = name.strip()
        value = value.strip()
        # Skip cookie attributes
        if name.lower() in {"path", "domain", "expires", "max-age", "secure", "httponly", "samesite"}:
            continue
        if not name:
            continue
        cookies.append({
            "name": name,
            "value": value,
            "domain": "." + domain,
            "path": "/",
            "httpOnly": False,
            "secure": True,
        })
    return cookies


def sanitize_lab_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\\/:\*\?\"<>\|\r\n\t]+", "_", name)  # replace path/invalid chars
    name = re.sub(r"\s+", "_", name)  # collapse spaces
    return name or "lab"

# -----------------------------
# DynamoDB helpers
# Table: CloudLabsFeedbackTable
# Partition key (PK): lab_id  e.g. "10370001/4504598964207616/6319033399771136"
# Additional attributes stored per item:
#   lab_display_name  (str)  — human-readable name from [Task 1]
#   lab_s3_name       (str)  — sanitized S3 key prefix
#   root_url          (str)  — canonical editor URL
#   extracted_at      (str)  — ISO-8601 timestamp of last extraction
# -----------------------------
DYNAMO_TABLE = "CloudLabsFeedbackTable"

@st.cache_resource(show_spinner=False)
def get_dynamodb():
    """Return a boto3 DynamoDB resource using the same AWS credentials as S3."""
    if boto3 is None:
        return None
    aws = st.secrets.get("aws", {})
    region = aws.get("region") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    access_key = aws.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = aws.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = aws.get("session_token") or os.getenv("AWS_SESSION_TOKEN")
    if access_key and secret_key:
        session = boto3.session.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region,
        )
    else:
        session = boto3.session.Session(region_name=region)
    return session.resource("dynamodb")


def _lab_id_from_url(url: str) -> Optional[str]:
    """
    Extract the canonical lab ID "id1/id2/id3" from any Educative URL.
    Works for both published and editor link formats.
    """
    ids = parse_editor_ids(url or "")
    if ids:
        return "/".join(ids)
    return None


def dynamo_get_lab(lab_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a lab record from DynamoDB by its lab_id.
    Returns the item dict, or None if not found / DynamoDB unavailable.
    """
    try:
        ddb = get_dynamodb()
        if ddb is None:
            return None
        table = ddb.Table(DYNAMO_TABLE)
        resp = table.get_item(Key={"lab_id": lab_id})
        return resp.get("Item")
    except Exception as e:
        print(f"[DynamoDB] get_lab error: {e}")
        return None


def dynamo_put_lab(lab_id: str, lab_display_name: str, lab_s3_name: str, root_url: str):
    """
    Insert or overwrite a lab record in DynamoDB.
    Always updates extracted_at to now.
    """
    try:
        ddb = get_dynamodb()
        if ddb is None:
            return
        table = ddb.Table(DYNAMO_TABLE)
        table.put_item(Item={
            "lab_id":           lab_id,
            "lab_display_name": lab_display_name,
            "lab_s3_name":      lab_s3_name,
            "root_url":         root_url,
            "extracted_at":     datetime.utcnow().isoformat() + "Z",
        })
    except Exception as e:
        print(f"[DynamoDB] put_lab error: {e}")


def dynamo_lab_exists(lab_id: str) -> bool:
    """Return True if a record for lab_id exists in DynamoDB."""
    return dynamo_get_lab(lab_id) is not None

# -----------------------------
# Report persistence helpers
# Reports are stored in S3 under:
#   analysis_reports/<lab_s3_name>_<user_id>_<attempt_id>_error_analysis.md
#   analysis_reports/<lab_s3_name>_<user_id>_<attempt_id>_compliance.md
# -----------------------------
def _report_key(lab_name: str, user_id: str, attempt_id: str, report_type: str,
                iam_username: str = "") -> str:
    """Build the S3 key for a saved analysis report."""
    lab = sanitize_lab_name(lab_name)
    uid = (user_id    or "unknown").strip()
    aid = (attempt_id or "unknown").strip()
    iam_suffix = f"_iam_{iam_username.strip()}" if (iam_username or "").strip() else ""
    return f"analysis_reports/{lab}_{uid}_{aid}{iam_suffix}_{report_type}.md"

def save_report_to_s3(s3, bucket: str, key: str, text: str):
    """Save a markdown report string to S3."""
    try:
        s3_put_text(s3, bucket, key, text, "text/markdown")
    except Exception as e:
        print(f"[report] save failed {key}: {e}")

def load_report_from_s3(s3, bucket: str, key: str) -> Optional[str]:
    """Load a markdown report from S3. Returns None if not found."""
    data = s3_get_bytes(s3, bucket, key)
    return data.decode("utf-8", errors="replace") if data else None

def chunk_logs(logs, chunk_size=1000):
    """Chunk logs for processing large files efficiently."""
    for i in range(0, len(logs), chunk_size):
        yield logs[i:i + chunk_size]

def _fmt_size(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

# -----------------------------
# S3 helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_s3() -> Tuple[Optional[str], Optional[object]]:
    """Create and cache a boto3 S3 client from secrets.toml. Returns (bucket, client)."""
    if boto3 is None:
        st.error("boto3 is not installed. Run: pip install boto3")
        return None, None
    aws = st.secrets.get("aws", {})
    bucket = aws.get("bucket")
    region = aws.get("region") or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
    access_key = aws.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = aws.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
    session_token = aws.get("session_token") or os.getenv("AWS_SESSION_TOKEN")

    if access_key and secret_key:
        session = boto3.session.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token,
            region_name=region,
        )
    else:
        session = boto3.session.Session(region_name=region)

    s3 = session.client("s3", config=BotoConfig(s3={"addressing_style": "virtual"}))
    return bucket, s3


def s3_put_text(s3, bucket: str, key: str, text: str, content_type: str):
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"), ContentType=content_type)

def s3_put_json(s3, bucket: str, key: str, obj: dict | list):
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(obj, indent=2).encode("utf-8"), ContentType="application/json")

def s3_get_bytes(s3, bucket: str, key: str) -> Optional[bytes]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise

def s3_object_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
# --- NEW HELPERS: load selected S3 files into session state ---------
def _load_text_to_state(s3, bucket: str, key: Optional[str], state_key: str):
    if not key:
        return
    data = s3_get_bytes(s3, bucket, key)
    if not data:
        return
    st.session_state[state_key] = data.decode("utf-8", errors="replace")

def _load_json_to_state(s3, bucket: str, key: Optional[str], state_key: str):
    if not key:
        return
    data = s3_get_bytes(s3, bucket, key)
    if not data:
        return
    try:
        st.session_state[state_key] = json.loads(data.decode("utf-8", errors="replace"))
    except Exception:
        # Fallback: keep raw string so user can see what's wrong
        st.session_state[state_key] = data.decode("utf-8", errors="replace")

def hydrate_selected_lab_assets_into_session(bucket: str, s3, sel: Dict[str, Optional[str]]):
    """
    Loads currently selected lab files from S3 into session:
      - lab_content            (from content_docs/<lab>.txt)
      - policy_json            (from policies/<lab>.json)
      - cloudformation_json    (from prescript/<lab>.json)
    """
    _load_text_to_state(s3, bucket, sel.get("content_key"), "lab_content")
    _load_json_to_state(s3, bucket, sel.get("policy_key"), "policy_json")
    _load_json_to_state(s3, bucket, sel.get("prescript_key"), "cloudformation_json")


@st.cache_data(show_spinner=False, max_entries=1, ttl=5)
def s3_list_labs(bucket: str, _s3) -> List[str]:
    """List lab names based on objects in content_docs/ (keys ending with .txt)."""
    labs: Set[str] = set()
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="content_docs/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".txt"):
                name = pathlib.Path(key).name[:-4]  # strip .txt
                labs.add(name)
    return sorted(labs)

@st.cache_data(show_spinner=False, ttl=5)
def s3_list_keys(bucket: str, _s3, prefix: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            items.append({
                "Key": obj["Key"],
                "Size": obj.get("Size", 0),
                "LastModified": obj.get("LastModified", ""),
                "ETag": obj.get("ETag", "")
            })
    items.sort(key=lambda x: x["Key"])
    return items

def s3_head(s3, bucket: str, key: str) -> Optional[Dict[str, str]]:
    try:
        return s3.head_object(Bucket=bucket, Key=key)
    except ClientError:
        return None

def view_s3_file_inline(s3, bucket: str, key: str, *, preview_chars: int = 2000):
    """Inline viewer: auto JSON pretty print or show text. Falls back to raw bytes length."""
    data = s3_get_bytes(s3, bucket, key)
    if data is None:
        st.warning("File not found.")
        return
    counter = getattr(view_s3_file_inline, '_counter', 0) + 1
    view_s3_file_inline._counter = counter
    ui_key_base = f"view_{hash(key)}_{counter}"

    head = s3_head(s3, bucket, key) or {}
    lm = head.get("LastModified", "")
    etag = head.get("ETag", "")
    size = head.get("ContentLength", len(data))
    st.caption(f"{key} • LastModified: {lm} • Size: {_fmt_size(size)} • ETag: {etag}")

    ext = pathlib.Path(key).suffix.lower()
    if ext == ".json":
        try:
            st.json(json.loads(data.decode("utf-8", errors="replace")), key=f"{ui_key_base}_json")
        except Exception:
            st.text_area("Raw JSON", value=data.decode("utf-8", errors="replace")[:preview_chars], height=300, key=f"{ui_key_base}_raw_json")
    elif ext in [".txt", ".log", ".md"]:
        max_chars = min(len(data), 200_000)
        n = st.slider(
            f"Preview characters for {pathlib.Path(key).name}",
            500,
            max(1000, max_chars),
            min(preview_chars, max_chars),
            step=500,
            key=f"{ui_key_base}_slider"
        )
        st.text_area("Preview", value=data.decode("utf-8", errors="replace")[:n], height=400, key=f"{ui_key_base}_preview")
    else:
        st.text_area("Raw preview", value=data[:preview_chars].decode("utf-8", errors="replace"), height=300, key=f"{ui_key_base}_raw")

    st.download_button("Download file", data=data, file_name=pathlib.Path(key).name, key=f"{ui_key_base}_download")


def s3_prefix_browser_ui(bucket: str, s3):
    st.markdown("#### S3 Browser")
    choice = st.radio(
        "Choose a prefix to browse",
        options=["content_docs/", "policies/", "prescript/", "execution_logs/", "resource_configs/", "Custom…"],
        horizontal=True
    )
    if choice == "Custom…":
        prefix = st.text_input("Enter custom prefix", value="content_docs/")
    else:
        prefix = choice

    cols = st.columns([3,1])
    with cols[0]:
        items = s3_list_keys(bucket, s3, prefix)
        if not items:
            st.info("No objects under this prefix.")
            return
        keys = [it["Key"] for it in items]
        key = st.selectbox("Select a file", options=keys, index=0)
    with cols[1]:
        st.metric("Objects", len(items))

    if key:
        view_s3_file_inline(s3, bucket, key)

def s3_view_lab_triplet_ui(bucket: str, s3, lab_choice: str):
    """Show inline viewers for content, policy, prescript of a given lab if present."""
    content_key = f"content_docs/{lab_choice}.txt"
    policy_key  = f"policies/{lab_choice}.json"
    presc_key   = f"prescript/{lab_choice}.json"

    st.markdown("### View files for this lab")

    if s3_object_exists(s3, bucket, content_key):
        st.markdown("**Content (TXT)**")
        view_s3_file_inline(s3, bucket, content_key, preview_chars=3000)
    else:
        st.caption("No content file found.")

    col1, col2 = st.columns(2)
    with col1:
        if s3_object_exists(s3, bucket, policy_key):
            st.markdown("**Policy (JSON)**")
            view_s3_file_inline(s3, bucket, policy_key)
        else:
            st.caption("No policy JSON.")

    with col2:
        if s3_object_exists(s3, bucket, presc_key):
            st.markdown("**Prescript (JSON)**")
            view_s3_file_inline(s3, bucket, presc_key)
        else:
            st.caption("No prescript JSON.")

    st.session_state["selected_lab_files"] = {
        "lab": lab_choice,
        "content_key": content_key if s3_object_exists(s3, bucket, content_key) else None,
        "policy_key": policy_key if s3_object_exists(s3, bucket, policy_key) else None,
        "prescript_key": presc_key if s3_object_exists(s3, bucket, presc_key) else None,
    }
    if st.button("Use these files for analysis", key=f"use_triplet_{lab_choice}"):
        hydrate_selected_lab_assets_into_session(bucket, s3, st.session_state["selected_lab_files"])
        st.success("Loaded lab content, policy JSON, and prescript (CloudFormation) into the workspace.")

# -----------------------------
# CloudTrail Log Processing
# -----------------------------
def load_and_process_logs(uploaded_file):
    """Process uploaded CloudTrail logs with progress tracking."""
    if uploaded_file is not None:
        try:
            progress_text = "Reading file..."
            progress_bar = st.progress(0)
            status_text = st.empty()

            file_size = getattr(uploaded_file, 'size', 0)
            if not file_size:
                # Handle cases where size is not available
                if isinstance(uploaded_file, bytes):
                    file_content = uploaded_file.decode("utf-8")
                elif isinstance(uploaded_file, str):
                    file_content = uploaded_file
                else:
                    uploaded_file.seek(0)
                    file_content = uploaded_file.read().decode("utf-8")
            else:
                chunk_size = 1024 * 1024
                file_content = []
                uploaded_file.seek(0)
                bytes_read = 0
                while True:
                    chunk = uploaded_file.read(chunk_size)
                    if not chunk:
                        break
                    file_content.append(chunk)
                    bytes_read += len(chunk)
                    progress = min(1.0, bytes_read / file_size)
                    progress_bar.progress(progress)
                    status_text.text(f"Reading file: {bytes_read}/{file_size} bytes ({progress:.1%})")
                    time.sleep(0.01)
                file_content = b''.join(file_content).decode("utf-8")

            status_text.text("Parsing JSON...")
            data = json.loads(file_content)
            events = data.get("events", [])
            total_events = len(events)
            status_text.text(f"Processing {total_events} events...")
            progress_bar.progress(0)

            organized_logs = defaultdict(list)
            processed_events = 0

            for chunk in chunk_logs(events):
                try:
                    sorted_chunk = sorted(
                        chunk,
                        key=lambda x: datetime.fromisoformat(x.get("event_time", "").replace("Z", "+00:00"))
                    )
                except:
                    st.warning("Some events have invalid timestamps. Using original order.")
                    sorted_chunk = chunk

                for event in sorted_chunk:
                    event_source = event.get("event_source", "unknown.source")
                    organized_logs[event_source].append(event)
                    processed_events += 1
                    if processed_events % 100 == 0:
                        progress = min(1.0, processed_events / total_events)  # Ensure progress doesn't exceed 1.0
                        progress_bar.progress(progress)
                        status_text.text(f"Processing events: {processed_events}/{total_events} ({progress:.1%})")
                        time.sleep(0.01)

            progress_bar.progress(1.0)
            status_text.text(f"Completed processing {total_events} events!")

            for source in organized_logs:
                try:
                    organized_logs[source].sort(
                        key=lambda x: datetime.fromisoformat(x.get("event_time", "").replace("Z", "+00:00"))
                    )
                except:
                    st.warning(f"Some events for {source} have invalid timestamps. Using original order.")

            return organized_logs

        except json.JSONDecodeError:
            st.error("Error: Invalid JSON.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    return None

def display_log_summary(organized_logs):
    """Display summary statistics for processed logs."""
    total_events = sum(len(logs) for logs in organized_logs.values())
    st.markdown("### ✨ Log Summary")
    st.write(f"**Total Event Sources:** {len(organized_logs)}")
    st.write(f"**Total Events:** {total_events}")

def display_event_details(log):
    """Display detailed information for a single CloudTrail event."""
    cloud_trail_event = log.get('cloud_trail_event', {})
    event_error_code = cloud_trail_event.get('errorCode')
    event_error_message = cloud_trail_event.get('errorMessage')

    cols = st.columns(3)
    with cols[0]:
        st.write("**Event Name:**", log.get('event_name', 'N/A'))
    with cols[1]:
        st.write("**Event Time:**", log.get('event_time', 'N/A'))
    with cols[2]:
        status_icon = "❌ Failed" if event_error_code else "✅ Success"
        status_color = "red" if event_error_code else "green"
        st.markdown(f"**Status:** <span style='color:{status_color}'>{status_icon}</span>", unsafe_allow_html=True)

    st.write("**Event ID:**", log.get('eventID', 'N/A'))
    user_identity = log.get('userIdentity', {})
    st.write("**User Identity:**")
    st.write(f"- **Type:** {user_identity.get('type', 'N/A')}")
    st.write(f"- **Username:** {user_identity.get('userName', user_identity.get('principalId', 'N/A'))}")
    st.write(f"- **Account ID:** {user_identity.get('accountId', 'N/A')}")
    st.write(f"- **ARN:** {user_identity.get('arn', 'N/A')}")
    st.write("**Source IP Address:**", cloud_trail_event.get('sourceIPAddress', 'N/A'))
    st.write("**User Agent:**", cloud_trail_event.get('userAgent', 'N/A'))

    if event_error_code:
        st.error(f"**Error Code:** {event_error_code}")
        st.error(f"**Error Message:** {event_error_message if event_error_message else 'Unknown error'}")

    if cloud_trail_event.get('requestParameters'):
        st.write("**Request Parameters:**")
        st.json(cloud_trail_event['requestParameters'])

    if cloud_trail_event.get('responseElements'):
        st.write("**Response Elements:**")
        st.json(cloud_trail_event['responseElements'])

def display_log_details(organized_logs):
    """Display detailed logs with filtering and pagination."""
    st.markdown("### 📈 Log Details by Event Source")

    service_names = list(organized_logs.keys())
    if service_names:
        # Use keys to track changes and reset pagination
        selected_service = st.selectbox("Select Service", service_names, key="service_select")

        service_logs = organized_logs[selected_service]

        status_filter = st.selectbox("Filter by Status", ["All", "Successful Calls", "Failed Calls"], key="status_select")

        event_names = sorted({log.get("event_name", "N/A") for log in service_logs})
        selected_event_names = st.multiselect(
            "Filter by Event Name(s)",
            options=event_names,
            default=event_names,
            key="eventname_multiselect"
        )

        # --- Reset pagination if any filter changed ---
        if (
            st.session_state.get("last_service") != selected_service or
            st.session_state.get("last_status") != status_filter or
            st.session_state.get("last_eventnames") != selected_event_names
        ):
            st.session_state["current_page"] = 1

        st.session_state["last_service"] = selected_service
        st.session_state["last_status"] = status_filter
        st.session_state["last_eventnames"] = selected_event_names

        # --- Apply filters ---
        filtered_logs = []
        for log in service_logs:
            cloud_trail_event = log.get('cloud_trail_event', {})
            event_error_code = cloud_trail_event.get('errorCode')

            matches_status = (
                status_filter == "All" or
                (status_filter == "Successful Calls" and not event_error_code) or
                (status_filter == "Failed Calls" and event_error_code)
            )
            matches_event_name = log.get("event_name", "N/A") in selected_event_names

            if matches_status and matches_event_name:
                filtered_logs.append(log)

        if not filtered_logs:
            st.info(f"No logs found for the selected filters.")
            return

        logs_per_page = st.slider("Events per page", 5, 50, 10, key="page_size_slider")
        total_pages = (len(filtered_logs) + logs_per_page - 1) // logs_per_page

        # Get or correct current page
        current_page = st.session_state.get('current_page', 1)
        if current_page > total_pages and total_pages > 0:
            current_page = total_pages
        elif total_pages == 0:
            current_page = 0

        page = st.number_input("Page", 1, max(1, total_pages), current_page, key="page_number_input") - 1
        st.session_state["current_page"] = page + 1

        start_idx = page * logs_per_page
        end_idx = min(start_idx + logs_per_page, len(filtered_logs))
        st.write(f"Showing events {start_idx + 1} to {end_idx} of {len(filtered_logs)}")

        for i, log in enumerate(filtered_logs[start_idx:end_idx], start=start_idx + 1):
            with st.expander(f"Event {i}: {log.get('event_name', 'N/A')}"):
                display_event_details(log)

# -----------------------------
# Gemini AI Analysis
# -----------------------------
async def analyze_lab_with_gemini(organized_logs, lab_spec_content):
    """Analyze lab issues using Gemini AI with failure logs."""
    st.info("Sending data to Model for analysis...")
    try:
        if "openaikey" not in st.secrets:
            st.error("OpenAI API key not found.")
            return

        genai.configure(api_key=st.secrets["openaikey"].get("key"))
        model = "gpt-4.1"

        summarized_logs = {}
        for source, events in organized_logs.items():
            error_events = [event for event in events if event.get('cloud_trail_event', {}).get('errorCode')]
            # Remove unnecessary fields from each event
            for event in error_events:
                if 'cloud_trail_event' in event:
                    event['cloud_trail_event'].pop('userAgent', None)
                    event['cloud_trail_event'].pop('readOnly', None)
                    # also drop where event_source has "guardduty", "devops-guru" or health in it
                    if "guardduty" in source or "devops-guru" in source or "health" or "ce" or "cost-optimization" or "notifications" or "cost-optimization" or "freetier" in source:
                        error_events.remove(event)
            summarized_logs[source] = error_events
        
        formatted_cloudtrail_logs = json.dumps(summarized_logs, indent=2)

        formatted_lab_specifications = lab_spec_content
        system_msg = (
            "You are an expert CloudTrail log analyst. "
            "Analyze only resources specified in the lab requirements and provide concise, actionable findings."
        )

        prompt = f"""
These are the events in the cloudtrail logs of a user's execution of the cloudlab.
You are to analyze only the logs that are related to the lab specifications. And use these logs to identify what issues the user might have encountered while attempting the lab and configuring these resources. Only analyze the logs for the resources specified.
These are the specific resources that are required in the lab:
Lab Specifications:
```json
{formatted_lab_specifications}
```
If there are some configurations for a resource that are not specified in the Lab specifications, you don't need to highlight them. If these configurations were required in the lab, these would have been specified there. Please provide a concise analysis, highlighting:
1. Summary of Issues (if any)
2. Affected Resources (if any) (only if the resource is specified in the lab specifications)
3. Troubleshooting Steps (if any)

CloudTrail Error Events Summary:
```json
{formatted_cloudtrail_logs}
```


""".strip()
        try:
            with st.spinner("Calling GPT-4.1..."):
                client = _openai_client()
                resp = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
            out = (resp.choices[0].message.content or "").strip()
            if not out:
                print("In if not out stage")
                st.warning("No response from GPT-4.1.")
                return
            else:
                print ("in else case and should print response",out)
                st.session_state["gemini_lab_analysis"] = out
                st.success("Resource creation compliance analysis generated with GPT-4.1.")
        except Exception as e:
            print("Error changing the model \n", e)
            st.error(f"GPT API not responding: {e}")

        # response = model.generate_content(prompt)
        # if response.text:
        #     st.session_state["gemini_lab_analysis"] = response.text
        # else:
        #     st.warning("No response from Gemini.")
    except Exception as e:
        st.error(f"API error: {e}")


async def analyze_resource_creation(organized_logs, lab_spec_content):
    """Analyze resource creation compliance using Gemini AI."""
    st.info("Analyzing resource creation compliance...")
    try:
        if "openaikey" not in st.secrets:
            st.error("OpenAI API key not found.")
            return

        genai.configure(api_key=st.secrets["openaikey"].get("key"))
        model = "gpt-4.1"

        creation_logs = {
            source: [event for event in events if any(x in event.get("event_name", "") for x in ("Create", "Authorize","Run","Publish","Put","Register","Attach", "Update"))]
            for source, events in organized_logs.items()
        }

        formatted_creation_logs = json.dumps(creation_logs, indent=2)

        system_msg = (
            "You are an AWS resource configuration expert. "
            "Given lab requirements and CloudTrail creation events, perform a compliance analysis."
        )

#You are an AWS resource configuration expert. Your task is to a
        creation_prompt = f"""
Analyze if the resources were created according to the lab specifications.

Lab Required Resources and Configurations:
```json
{lab_spec_content}
```

Actual Resource Creation Events from CloudTrail:
```json
{formatted_creation_logs}
```

Please perform a detailed compliance analysis:

1. Resource Creation Verification
   - For each resource specified in the lab requirements:
     * Check if it was created (found in CloudTrail logs)
     * Verify if all required configurations were applied correctly
     * Identify any missing or incorrect configurations

2. Configuration Compliance
   - Compare the actual configurations used in creation events with the required specifications
   - List any discrepancies in:
     * Resource properties
     * Tags
     * Security settings
     * Other specified parameters

Please provide your analysis in the following format:

1. Resources Created Successfully
   - List each resource that was created correctly with all required configurations
   - Include the specific configurations that match the requirements

2. Resources with Configuration Issues
   - List resources where configurations don't match requirements
   - Specify which configurations are incorrect or missing
   - Include both the required and actual values

3. Missing Resources
   - List any required resources that were not found in the creation events
""".strip()
        try:
            with st.spinner("Calling GPT-4.1..."):
                client = _openai_client()
                resp = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": creation_prompt},
                    ],
                    temperature=0.2,
                )
            out = (resp.choices[0].message.content or "").strip()
            if not out:
                print("In if not out stage")
                st.warning("No response from GPT-4.1.")
                return
            else:
                print ("in else case and should print response",out)
                st.session_state["gemini_resource_analysis"] = out
                st.success("Resource creation compliance analysis generated with GPT-4.1.")
        except Exception as e:
            print("Error changing the model \n", e)

        # response = model.generate_content(creation_prompt)
        # if response.text:
        #     st.session_state["gemini_resource_analysis"] = response.text
        # else:
        #     st.warning("No response from Gemini.")
    except Exception as e:
        st.error(f"API error: {e}")

def cloudtrail_analysis_module():
    """Main CloudTrail analysis module — auto-uses data loaded by the auto-flow above."""
    # Initialize session state for persistent data FIRST
    if "cloudtrail_organized_logs" not in st.session_state:
        st.session_state.cloudtrail_organized_logs = None
    if "cloudtrail_lab_spec_content" not in st.session_state:
        st.session_state.cloudtrail_lab_spec_content = None
    if "cloudtrail_selected_log_file" not in st.session_state:
        st.session_state.cloudtrail_selected_log_file = None
    if "cloudtrail_selected_config_file" not in st.session_state:
        st.session_state.cloudtrail_selected_config_file = None
    if "cloudtrail_active_tab" not in st.session_state:
        st.session_state.cloudtrail_active_tab = "Summary"

    st.header("CloudTrail Log Analyzer")

    # Get S3 connection
    bucket, s3 = get_s3()
    if not bucket or not s3:
        st.error("Missing S3 setup. Add [aws] bucket/region/credentials to secrets.toml.")
        return

    # If the auto-flow already populated logs, show them directly — no manual selection needed
    if st.session_state.cloudtrail_organized_logs and st.session_state.cloudtrail_lab_spec_content:
        # Data already loaded — skip straight to analysis tabs below
        pass
    else:
        # Fallback manual selection (for power users / re-analysis without going through the flow again)
        with st.expander("📂 Manually load files from S3 or upload (optional)", expanded=not bool(st.session_state.cloudtrail_organized_logs)):
            file_source = st.radio(
                "File source",
                options=["Select from S3", "Upload from device"],
                horizontal=True,
                key="cloudtrail_file_source"
            )

            if file_source == "Select from S3":
                execution_logs = s3_list_keys(bucket, s3, "execution_logs/")
                if execution_logs:
                    log_files = [item["Key"] for item in execution_logs]
                    selected_log_file = st.selectbox("CloudTrail log file", options=log_files, key="s3_log_select")
                else:
                    selected_log_file = None
                    st.info("No execution logs found in S3.")

                # Load logs button
                if selected_log_file and st.button("Load and Process Logs", key="load_logs_btn"):
                    with st.spinner("Loading logs from S3..."):
                        log_data = s3_get_bytes(s3, bucket, selected_log_file)
                        if log_data:
                            st.session_state['log_content'] = log_data.decode('utf-8')
                            class _MockFile2:
                                def __init__(self, data): self.data = data; self.position = 0
                                def read(self, size=None):
                                    if size is None: chunk = self.data[self.position:]; self.position = len(self.data); return chunk
                                    end_pos = min(self.position + size, len(self.data)); chunk = self.data[self.position:end_pos]; self.position = end_pos; return chunk
                                def seek(self, pos): self.position = pos
                            st.session_state.cloudtrail_organized_logs = load_and_process_logs(_MockFile2(log_data))
                            st.session_state.cloudtrail_selected_log_file = selected_log_file
                            st.success(f"Loaded logs from s3://{bucket}/{selected_log_file}")
                        else:
                            st.error("Failed to load log file from S3")

                resource_configs = s3_list_keys(bucket, s3, "resource_configs/")
                if resource_configs:
                    config_files = [item["Key"] for item in resource_configs]
                    selected_config_file = st.selectbox("Resource config file", options=config_files, key="s3_config_select")
                    if selected_config_file and st.button("Load Resource Config", key="load_config_btn"):
                        config_data = s3_get_bytes(s3, bucket, selected_config_file)
                        if config_data:
                            lab_name_from_cfg = selected_config_file.split('/')[-1].replace('_resource_config.json', '')
                            lab_content_bytes = s3_get_bytes(s3, bucket, f"content_docs/{lab_name_from_cfg}.txt")
                            if lab_content_bytes:
                                st.session_state['lab_content'] = lab_content_bytes.decode('utf-8', errors='replace')
                            st.session_state.cloudtrail_lab_spec_content = config_data.decode("utf-8", errors="replace")
                            st.session_state.cloudtrail_selected_config_file = selected_config_file
                            st.success(f"Loaded resource config from s3://{bucket}/{selected_config_file}")
                        else:
                            st.error("Failed to load resource config from S3")
                else:
                    st.info("No resource configs found in S3.")

            else:  # Upload from device
                uploaded_file = st.file_uploader("Upload CloudTrail JSON Log File", type="json", key="cloudtrail_uploader")
                if uploaded_file:
                    with st.spinner("Processing logs..."):
                        st.session_state.cloudtrail_organized_logs = load_and_process_logs(uploaded_file)
                lab_spec_file = st.file_uploader("Upload Lab Specification (JSON)", type="json", key="lab_spec")
                if lab_spec_file:
                    try:
                        st.session_state.cloudtrail_lab_spec_content = lab_spec_file.read().decode("utf-8")
                        st.success("Lab specifications loaded.")
                    except Exception:
                        st.error("Invalid lab specification JSON.")

    # Status indicators
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.cloudtrail_organized_logs:
            st.success(f"✅ Logs loaded: {st.session_state.cloudtrail_selected_log_file or 'Uploaded file'}")
        else:
            st.info("📋 No logs loaded")

    with col2:
        if st.session_state.cloudtrail_lab_spec_content:
            st.success(f"✅ Config loaded: {st.session_state.cloudtrail_selected_config_file or 'Uploaded file'}")
        else:
            st.info("📋 No resource config loaded")

    # Analysis section (only show if we have logs) - OUTSIDE COLUMNS
    if st.session_state.cloudtrail_organized_logs:
        st.success("Logs processed successfully!")
        
        # Create tabs with state management
        tab_names = ["Summary", "Detailed Logs", "Errors Analysis in Lab", "Resource Compliance"]
        
        st.markdown("### Analysis Options")
        # Use radio buttons to maintain tab state
        selected_tab = st.radio(
            "Select Analysis Tab",
            options=tab_names,
            horizontal=True,
            key="cloudtrail_tab_selector",
            index=tab_names.index(st.session_state.cloudtrail_active_tab) if st.session_state.cloudtrail_active_tab in tab_names else 0
        )
        
        # Update active tab in session state
        st.session_state.cloudtrail_active_tab = selected_tab
        
        st.divider()
        
        # Display content based on selected tab
        if selected_tab == "Summary":
            display_log_summary(st.session_state.cloudtrail_organized_logs)

        elif selected_tab == "Detailed Logs":
            display_log_details(st.session_state.cloudtrail_organized_logs)

        elif selected_tab == "Errors Analysis in Lab":
            st.markdown("### 🔍 Error Analysis Results")
            if not st.session_state.cloudtrail_lab_spec_content:
                st.info("Resource config not yet loaded — complete the main flow above first.")
            else:
                analysis = st.session_state.get("gemini_lab_analysis")
                if analysis:
                    # Show cached report + optional re-run
                    saved_key = st.session_state.get("flow_error_analysis_key", "")
                    if saved_key:
                        st.caption(f"📄 Report saved to S3: `{saved_key}`")
                    st.markdown(analysis)
                    if st.button("🔄 Re-run error analysis", key="rerun_error_btn"):
                        st.session_state.pop("gemini_lab_analysis", None)
                        st.session_state["flow_error_analysis_done"] = False
                        st.session_state["flow_error_analysis_key"] = None
                        st.rerun()
                else:
                    # Not yet run — trigger manually if auto-flow hasn't fired yet
                    if st.button("▶ Run Error Analysis", key="analyze_lab_btn", type="primary"):
                        try:
                            asyncio.run(analyze_lab_with_gemini(
                                st.session_state.cloudtrail_organized_logs,
                                st.session_state.cloudtrail_lab_spec_content
                            ))
                            bucket2, s3_2 = get_s3()
                            report = st.session_state.get("gemini_lab_analysis", "")
                            if report and bucket2 and s3_2:
                                lab_n  = st.session_state.get("flow_lab_name", "unknown")
                                uid    = st.session_state.get("flow_user_id", "unknown")
                                aid    = st.session_state.get("flow_selected_attempt", "unknown")
                                iam_u  = st.session_state.get("flow_iam_username", "")
                                ekey   = _report_key(lab_n, uid, aid, "error_analysis", iam_u)
                                save_report_to_s3(s3_2, bucket2, ekey, report)
                                st.session_state["flow_error_analysis_key"]  = ekey
                                st.session_state["flow_error_analysis_done"] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")

        elif selected_tab == "Resource Compliance":
            st.markdown("### 🛠 Resource Compliance Results")
            if not st.session_state.cloudtrail_lab_spec_content:
                st.info("Resource config not yet loaded — complete the main flow above first.")
            else:
                compliance = st.session_state.get("gemini_resource_analysis")
                if compliance:
                    saved_key = st.session_state.get("flow_compliance_key", "")
                    if saved_key:
                        st.caption(f"📄 Report saved to S3: `{saved_key}`")
                    st.markdown(compliance)
                    if st.button("🔄 Re-run compliance check", key="rerun_compliance_btn"):
                        st.session_state.pop("gemini_resource_analysis", None)
                        st.session_state["flow_compliance_done"] = False
                        st.session_state["flow_compliance_key"] = None
                        st.rerun()
                else:
                    if st.button("▶ Run Compliance Check", key="analyze_compliance_btn", type="primary"):
                        try:
                            asyncio.run(analyze_resource_creation(
                                st.session_state.cloudtrail_organized_logs,
                                st.session_state.cloudtrail_lab_spec_content
                            ))
                            bucket2, s3_2 = get_s3()
                            report = st.session_state.get("gemini_resource_analysis", "")
                            if report and bucket2 and s3_2:
                                lab_n = st.session_state.get("flow_lab_name", "unknown")
                                uid   = st.session_state.get("flow_user_id", "unknown")
                                aid   = st.session_state.get("flow_selected_attempt", "unknown")
                                iam_u = st.session_state.get("flow_iam_username", "")
                                ckey  = _report_key(lab_n, uid, aid, "compliance", iam_u)
                                save_report_to_s3(s3_2, bucket2, ckey, report)
                                st.session_state["flow_compliance_key"]  = ckey
                                st.session_state["flow_compliance_done"] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Compliance check failed: {e}")
    else:
        st.info("Please select or upload CloudTrail logs to begin analysis.")

# -----------------------------
# Generic crawler fallback
# -----------------------------
def visible_text_only(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "form", "iframe"]):
        tag.decompose()
    for tag in soup.find_all(style=True):
        style = (tag.get("style", "") or "").replace(" ", "").lower()
        if "display:none" in style:
            tag.decompose()
    return "\n".join([s for s in soup.stripped_strings])

def crawl_generic(root_url: str, session: requests.Session, max_pages: int, same_path_only: bool, delay: float) -> Dict[str, str]:
    visited: Set[str] = set()
    queue: List[str] = [root_url]
    results: Dict[str, str] = {}

    root = urlparse(root_url)

    def in_scope(candidate: str) -> bool:
        c = urlparse(candidate)
        if c.netloc != root.netloc:
            return False
        if same_path_only and not c.path.startswith(root.path.rstrip("/") + "/") and c.path != root.path:
            return False
        return True

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code >= 400:
                continue
            soup = BeautifulSoup(resp.text, "lxml")
            results[url] = visible_text_only(soup)
            for a in soup.find_all("a", href=True):
                nxt = normalize_url(url, a["href"]) or ""
                if nxt and nxt not in visited and nxt not in queue and in_scope(nxt):
                    queue.append(nxt)
            time.sleep(max(0.0, delay))
        except Exception:
            pass
    return results

# -----------------------------
# Lab Extraction Functions
# -----------------------------
def s3_write_outputs_to_folders(
    s3,
    bucket: str,
    lab_name: str,
    root_url: str,
    landing: TaskArtifact,
    tasks: List[TaskArtifact],
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Writes to S3:
      content_docs/<lab>.txt
      policies/<lab>.json    (landing-page code block #1, if present)
      prescript/<lab>.json   (landing-page code block #2, if present)
    Returns S3 KEYS: (content_key, policy_key_or_None, prescript_key_or_None)
    """
    lab = sanitize_lab_name(lab_name)

    # 1) content_docs/<lab>.txt
    out = io.StringIO()
    out.write("LAB CONTENT\n")
    out.write(f"Lab name: {lab_name}\n")
    out.write(f"Root URL: {root_url}\n")
    out.write(f"Total tasks: {len(tasks)}\n")
    out.write("=" * 80 + "\n\n")
    for i, t in enumerate(tasks, start=1):
        out.write(f"[Task {i}] {t.title}\n")
        out.write(f"URL: {t.url}\n")
        out.write("-" * 80 + "\n")
        if t.slate_text.strip():
            out.write(t.slate_text.strip() + "\n\n")
        else:
            out.write("(No Slate text extracted)\n\n")
    content_key = f"content_docs/{lab}.txt"
    s3_put_text(s3, bucket, content_key, out.getvalue(), "text/plain")

    # 2) policy (first ACE block)
    policy_key = None
    if landing and landing.code_blocks:
        if len(landing.code_blocks) >= 1 and landing.code_blocks[0].get("content"):
            policy_key = f"policies/{lab}.json"
            s3_put_text(s3, bucket, policy_key, landing.code_blocks[0]["content"], "application/json")

    # 3) prescript (second ACE block)
    prescript_key = None
    if landing and landing.code_blocks:
        if len(landing.code_blocks) >= 2 and landing.code_blocks[1].get("content"):
            prescript_key = f"prescript/{lab}.json"
            s3_put_text(s3, bucket, prescript_key, landing.code_blocks[1]["content"], "application/json")

    return content_key, policy_key, prescript_key

# -----------------------------
# Educative (Requests fallback)
# -----------------------------
EDU_SLATE_SELECTOR = 'div[role="textbox"][data-testid="slate-content-editable"]'
EDU_ACE_CONTENT_SELECTOR = 'div.ace_content'

def educative_collect_task_links_requests(landing_url: str, html: str, debug: bool = False) -> List[TaskLink]:
    soup = BeautifulSoup(html, "lxml")
    base = landing_url

    categories: List[Tuple[Tag, str]] = []
    for span in soup.find_all("span", class_="text-xl"):
        title_text = span.get_text(strip=True)
        container = span.find_parent("div")
        if container:
            categories.append((container, title_text))

    def derive_category(node: Tag) -> Optional[str]:
        parent = node
        while parent and getattr(parent, "name", None) != "body":
            for container, cat_title in categories:
                if container and (container in parent.parents):
                    return cat_title
            parent = getattr(parent, "parent", None)
        return None

    links: List[TaskLink] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "cloudlabeditor" in href:
            href_abs = normalize_url(base, href)
            title = a.get_text(strip=True)
            cat = derive_category(a)
            if href_abs:
                links.append(TaskLink(category=cat, title=title, href=href_abs))

    if not links:
        pattern = re.compile(r"/(?:editor/)?cloudlabeditor/[0-9]+/[0-9]+/[0-9]+/[0-9]+")
        matches = list(dict.fromkeys(pattern.findall(html)))
        for href in matches:
            href_abs = normalize_url(base, href)
            if href_abs:
                links.append(TaskLink(category=None, title="", href=href_abs))

    seen = set()
    deduped: List[TaskLink] = []
    for tl in links:
        key = (tl.title or "", tl.href)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tl)
    return deduped


def educative_extract_task_requests(html: str, url: str) -> TaskArtifact:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else url

    # Slate editor content
    slate_text_blocks: List[str] = []
    for editor in soup.select(EDU_SLATE_SELECTOR):
        slate_text_blocks.append(editor.get_text("\n", strip=True))
    slate_text = "\n".join([b for b in slate_text_blocks if b])

    # ACE editor blocks (DOM fallback; may be partial on virtualized editors)
    code_blocks: List[Dict[str, str]] = []
    for ace in soup.select(EDU_ACE_CONTENT_SELECTOR):
        text_layer = ace.find("div", class_=re.compile(r"ace_text-layer"))
        if text_layer:
            lines = [line.get_text("", strip=False) for line in text_layer.select("div.ace_line")]
            content = "\n".join(lines).rstrip()
            if content:
                code_blocks.append({"type": "ace", "content": content})

    return TaskArtifact(url=url, title=title, slate_text=slate_text, code_blocks=code_blocks)

# -----------------------------
# Playwright helpers (ACE extraction)
# -----------------------------
def pw_get_slate_text(page) -> str:
    try:
        return page.eval_on_selector_all(
            '[data-testid="slate-content-editable"]',
            'nodes => nodes.map(n => n.innerText).join("\\n\\n")'
        ) or ""
    except Exception:
        return ""

def pw_get_all_ace_values(page) -> List[str]:
    js = r"""
    () => {
      const results = [];
      const aceGlobal = window.ace;
      const editorEls = Array.from(document.querySelectorAll('.ace_editor'));
      if (aceGlobal && editorEls.length) {
        for (const el of editorEls) {
          try {
            const ed = aceGlobal.edit(el);
            if (ed && ed.getSession) {
              results.push(ed.getValue());
              continue;
            }
          } catch (e) { /* fall through */ }
          const layer = el.querySelector('.ace_text-layer');
          if (layer) {
            const lines = Array.from(layer.querySelectorAll('.ace_line')).map(l => l.textContent || '');
            results.push(lines.join('\n'));
          } else {
            results.push(el.textContent || '');
          }
        }
        return results;
      }
      const contents = Array.from(document.querySelectorAll('div.ace_content'));
      for (const root of contents) {
        const layer = root.querySelector('div.ace_text-layer');
        if (layer) {
          const lines = Array.from(layer.querySelectorAll('div.ace_line')).map(l => l.textContent || '');
          results.push(lines.join('\n'));
        } else {
          results.push(root.textContent || '');
        }
      }
      return results;
    }
    """
    try:
        vals = page.evaluate(js)
        return [ (v or "") for v in vals ]
    except Exception:
        try:
            return page.eval_on_selector_all(
                'div.ace_content',
                '''nodes => nodes.map(root => {
                    const layer = root.querySelector('div.ace_text-layer');
                    if (!layer) return root.textContent || '';
                    const lines = Array.from(layer.querySelectorAll('div.ace_line')).map(l => l.textContent || '');
                    return lines.join('\\n');
                })'''
            ) or []
        except Exception:
            return []

# -----------------------------
# Execution Logs helpers
# -----------------------------
EDITOR_IDS_RE = re.compile(r"/cloudlabeditor/(\d+)/(\d+)/(\d+)", re.IGNORECASE)
# Published link: /collection/page/<id1>/<id2>/<id3>/cloudlab  or  /collection/page/<id1>/<id2>/<id3>
PUBLISHED_IDS_RE = re.compile(r"/collection/page/(\d+)/(\d+)/(\d+)", re.IGNORECASE)

def normalize_lab_url_to_editor(url: str) -> str:
    """
    Accept both published and editor URLs and return a canonical editor URL.
    Published:  https://www.educative.io/collection/page/10370001/4618838700982272/5256267680710656/cloudlab
    Editor:     https://www.educative.io/editor/cloudlabeditor/10370001/4618838700982272/5256267680710656
    Returns the editor URL (or the original if already in editor format / unrecognized).
    """
    url = (url or "").strip()
    # Already an editor URL?
    if EDITOR_IDS_RE.search(url):
        return url
    # Try published format
    m = PUBLISHED_IDS_RE.search(url)
    if m:
        id1, id2, id3 = m.group(1), m.group(2), m.group(3)
        return f"https://www.educative.io/editor/cloudlabeditor/{id1}/{id2}/{id3}"
    return url

def parse_editor_ids(editor_url: str) -> Optional[Tuple[str, str, str]]:
    # Normalise first so published links also work
    normalised = normalize_lab_url_to_editor(editor_url or "")
    m = EDITOR_IDS_RE.search(normalised)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)

def build_events_api_url(editor_url: str, user_id: str, attempt_id: str, region: str,
                         iam_username: str = "") -> Optional[str]:
    ids = parse_editor_ids(editor_url)
    if not ids:
        return None
    id1, id2, id3 = ids
    region     = (region     or "").strip()
    user_id    = (user_id    or "").strip()
    attempt_id = (attempt_id or "").strip()
    if not (region and user_id and attempt_id):
        return None
    url = f"https://www.educative.io/api/cloud-lab/{id1}/{id2}/{id3}/events?user_id={user_id}&attempt_id={attempt_id}&region={region}"
    if (iam_username or "").strip():
        url += f"&iam_username={iam_username.strip()}"
    return url

def get_root_url_from_content_file(s3, bucket: str, lab: str) -> Optional[str]:
    key = f"content_docs/{lab}.txt"
    data = s3_get_bytes(s3, bucket, key)
    if not data:
        return None
    text = data.decode("utf-8", errors="replace")
    m = re.search(r"^Root URL:\s*(\S+)", text, flags=re.MULTILINE)
    return m.group(1) if m else None

def extract_lab_display_name_from_content(content_text: str) -> Optional[str]:
    """Extract the human-readable lab name from [Task 1] line in content file."""
    m = re.search(r"^\[Task 1\]\s+(.+)$", content_text, flags=re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None

def rename_lab_files_in_s3(s3, bucket: str, old_name: str, new_name: str) -> bool:
    """
    Rename all S3 files for a lab from old_name to new_name.
    Copies objects to new keys and deletes the originals.
    Returns True if any renames were performed.
    """
    prefixes_and_suffixes = [
        ("content_docs/", ".txt"),
        ("policies/", ".json"),
        ("prescript/", ".json"),
        ("resource_configs/", "_resource_config.json"),
    ]
    old_san = sanitize_lab_name(old_name)
    new_san = sanitize_lab_name(new_name)
    if old_san == new_san:
        return False
    renamed = False
    for prefix, suffix in prefixes_and_suffixes:
        old_key = f"{prefix}{old_san}{suffix}"
        new_key = f"{prefix}{new_san}{suffix}"
        if s3_object_exists(s3, bucket, old_key):
            try:
                s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": old_key}, Key=new_key)
                s3.delete_object(Bucket=bucket, Key=old_key)
                renamed = True
            except Exception as e:
                print(f"Rename failed {old_key} → {new_key}: {e}")
    # Also rename execution_logs which have a different pattern
    exec_prefix = f"execution_logs/{old_san}_"
    items = s3_list_keys(bucket, s3, exec_prefix)
    for item in items:
        old_key = item["Key"]
        new_key = old_key.replace(exec_prefix, f"execution_logs/{new_san}_", 1)
        try:
            s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": old_key}, Key=new_key)
            s3.delete_object(Bucket=bucket, Key=old_key)
            renamed = True
        except Exception as e:
            print(f"Rename exec log failed {old_key}: {e}")
    return renamed

def fetch_execution_logs(api_url: str, cookie_header: str) -> Dict:
    """Fetch JSON logs from events API. Uses Cookie header if provided."""
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    if cookie_header:
        cookie_val = cookie_header
        if ":" in cookie_val:
            k, v = cookie_val.split(":", 1)
            if k.strip().lower() == "cookie":
                cookie_val = v.strip()
        sess.headers["Cookie"] = cookie_val.strip()
    resp = sess.get(api_url, timeout=45)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"raw": resp.text}

def list_execution_logs_for_lab(s3, bucket: str, lab: str) -> List[str]:
    prefix = f"execution_logs/{sanitize_lab_name(lab)}_"
    items = s3_list_keys(bucket, s3, prefix)
    return [it["Key"] for it in items]
def build_history_api_url(editor_url: str, user_id: str) -> Optional[str]:
    """
    Convert editor URL to attempts history API:
      https://<host>/api/cloud-lab/<id1>/<id2>/<id3>/history?user_id=<user_id>
    """
    if not editor_url or not user_id:
        return None
    try:
        # Reuse your existing parse_editor_ids(editor_url)
        ids = parse_editor_ids(editor_url)
        if not ids or len(ids) != 3:
            return None
        id1, id2, id3 = ids
        parsed = urlparse(editor_url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or "www.educative.io"
        return f"{scheme}://{netloc}/api/cloud-lab/{id1}/{id2}/{id3}/history?user_id={quote_plus(user_id)}"
    except Exception:
        return None


def fetch_attempt_history(api_url: str, cookie_header: str, timeout: int = 25) -> Any:
    """
    Call the history endpoint and return parsed JSON. Raises with a clear message if not JSON.
    """
    headers = {"Cookie": cookie_header, "Accept": "application/json"}
    resp = requests.get(api_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        raise RuntimeError("History API did not return JSON.")
def parse_attempts_from_history(history_json: Any) -> List[Dict[str, Any]]:
    """
    Normalize the history payload into a list of attempts.
    We support a few common shapes and key names.
    """
    # Figure out where the attempts array lives
    if isinstance(history_json, list):
        raw = history_json
    elif isinstance(history_json, dict):
        raw = (
            history_json.get("history")
            or history_json.get("attempts")
            or history_json.get("data")
            or []
        )
    else:
        raw = []

    attempts: List[Dict[str, Any]] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        attempt_id = a.get("attempt_id") or a.get("attemptId") or a.get("id") or a.get("attempt")
        # region = a.get("region") or a.get("aws_region") or a.get("location")
        # status = a.get("status") or a.get("state")
        started = a.get("started_at") or a.get("start_time") or a.get("createdAt") or a.get("created_at")
        ended = a.get("ended_at") or a.get("end_time") or a.get("completedAt") or a.get("completed_at")
        # Keep any extra useful metadata if present
        extra = {
            k: v for k, v in a.items()
            if k not in {"attempt_id","attemptId","id","attempt","region","aws_region","location",
                         "status","state","started_at","start_time","createdAt","created_at",
                         "ended_at","end_time","completedAt","completed_at"}
        }
        attempts.append({
            "attempt_id": str(attempt_id) if attempt_id is not None else "",
            # "region": region or "",
            # "status": status or "",
            "started": started or "",
            "ended": ended or "",
            **({"meta": extra} if extra else {})
        })
    return attempts

# cloudtrail_section_ui removed — logic handled inside lab_extractor_module auto-flow

# -----------------------------
# OpenAI single-call extraction
# -----------------------------
def _openai_client() -> Optional[OpenAI]:
    key = (st.secrets.get("openaikey") or {}).get("key")
    if not key:
        st.error("OpenAI key not found in secrets.toml under [openaikey].key")
        return None
    return OpenAI(api_key=key)

RC_SYSTEM_PROMPT = """You are a careful extractor. Read the provided context and extract ONLY the cloud resources that a learner is instructed to CREATE or CONFIGURE.

Return ONLY a single valid JSON object with this EXACT shape:
{
  "Resources": [
    {
      "Resource Name": "<name>",
      "Service Type": "<AWS service>",
      "Configurations": { },
      "Dependencies": []
    }
  ]
}
If nothing is found, return {"Resources": []}.

Rules:
- Only include resources explicitly created/configured by the learner (not example names unless used).
- Prefer concrete fields; use short placeholders like "<unique_suffix>" sparingly.
- Dependencies should contain only names of other resources in this JSON.
- Do NOT add commentary or any text outside the JSON object.
"""

def call_openai_extract(client: OpenAI, context: str, question: str, *, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Single-call extraction using the entire context (no chunking).
    Tries the newer Responses API first; falls back to Chat Completions if unavailable.
    """
    model = (
        model
        or (st.secrets.get("openai", {}) or {}).get("model")
        or "gpt-5"
    )

    user_prompt = f"""Context:
{context}

Question: {question}

Return ONLY one JSON object with the required schema. No prose."""

    # Prefer Responses API
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": RC_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        )
        content = ""
        if getattr(resp, "output_text", None):
            content = resp.output_text
        else:
            # dig out text from chunks if needed
            chunks = []
            for item in getattr(resp, "output", []) or []:
                for p in getattr(item, "content", []) or []:
                    if getattr(p, "type", "") == "output_text":
                        chunks.append(getattr(p, "text", "") or "")
            content = "\n".join(chunks)
    except Exception:
        # Fallback: Chat Completions
        fallback_model = model if model != "gpt-5" else "gpt-4.1"
        resp = client.chat.completions.create(
            model=fallback_model,
            messages=[
                {"role": "system", "content": RC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""

    # Extract first JSON object from the model output
    m = re.search(r"({[\s\S]*})", content)
    if not m:
        return {"Resources": []}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {"Resources": []}

# -----------------------------
# Generic, service-agnostic normalizer
# -----------------------------
NULLISH_STRINGS = {"", "none", "null", "n/a", "not specified"}

def _is_nullish(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in NULLISH_STRINGS:
        return True
    if isinstance(val, (list, dict)) and len(val) == 0:
        return True
    return False

def _merge_lists(a: List[Any], b: List[Any]) -> List[Any]:
    out: List[Any] = []
    seen: Set[str] = set()
    for item in list(a) + list(b):
        key = json.dumps(item, sort_keys=True) if not isinstance(item, str) else item.strip()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out

def _merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-recursive merge: dicts recurse, lists dedupe, scalars prefer non-nullish existing, else new."""
    out = dict(a or {})
    for k, v in (b or {}).items():
        if k in out:
            if isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = _merge_dicts(out[k], v)
            elif isinstance(out[k], list) and isinstance(v, list):
                out[k] = _merge_lists(out[k], v)
            else:
                if _is_nullish(out[k]) and not _is_nullish(v):
                    out[k] = v
                # else keep existing
        else:
            out[k] = v
    return out

def _prune(obj: Any) -> Any:
    if isinstance(obj, dict):
        pruned = {k: _prune(v) for k, v in obj.items()}
        # remove keys that became nullish
        return {k: v for k, v in pruned.items() if not _is_nullish(v)}
    if isinstance(obj, list):
        return [ _prune(v) for v in obj if not _is_nullish(v) ]
    return obj

def normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Enforce schema, dedupe by (Resource Name, Service Type), merge configs/dependencies, prune empties."""
    out: Dict[str, Any] = {"Resources": []}
    resources = payload.get("Resources") if isinstance(payload, dict) else None
    if not isinstance(resources, list):
        return out

    seen: Dict[Tuple[str, str], int] = {}
    for r in resources:
        if not isinstance(r, dict):
            continue
        name = str(r.get("Resource Name", "")).strip()
        svc  = str(r.get("Service Type", "")).strip()
        if not name or not svc:
            continue

        cfg = r.get("Configurations")
        if not isinstance(cfg, dict):
            cfg = {}
        cfg = _prune(cfg)

        deps = r.get("Dependencies")
        if isinstance(deps, list):
            deps = [str(d).strip() for d in deps if str(d).strip()]
        elif _is_nullish(deps):
            deps = []
        else:
            deps = [str(deps).strip()]
        deps = sorted(set(deps))

        key = (name, svc)
        if key in seen:
            idx = seen[key]
            # merge configurations
            out["Resources"][idx]["Configurations"] = _merge_dicts(out["Resources"][idx]["Configurations"], cfg)
            # merge deps
            out["Resources"][idx]["Dependencies"] = sorted(set(out["Resources"][idx]["Dependencies"]) | set(deps))
        else:
            out["Resources"].append({
                "Resource Name": name,
                "Service Type": svc,
                "Configurations": cfg,
                "Dependencies": deps
            })
            seen[key] = len(out["Resources"]) - 1

    # Final prune to drop any remaining empties inside configs
    out["Resources"] = [
        {
            "Resource Name": r["Resource Name"],
            "Service Type": r["Service Type"],
            "Configurations": _prune(r.get("Configurations", {})) or {},
            "Dependencies": r.get("Dependencies", []),
        }
        for r in out["Resources"]
    ]
    return out

# -----------------------------
# Resource Config generation (single-call, full context)
# -----------------------------


def _load_text_from_s3(s3, bucket: str, key: str) -> Optional[str]:
    data = s3_get_bytes(s3, bucket, key)
    return data.decode("utf-8", errors="replace") if data else None

def _pick_latest_logs_key(bucket: str, s3, lab_name: str) -> Optional[str]:
    prefix = f"execution_logs/{sanitize_lab_name(lab_name)}_"
    items = s3_list_keys(bucket, s3, prefix)
    if not items:
        return None
    items.sort(key=lambda it: it.get("LastModified") or "", reverse=True)
    return items[0]["Key"]

def generate_resource_config_section(*, bucket: str, s3, lab_name: str):
    """
    Build context from the ENTIRE content_docs/<lab>.txt (plus optional logs),
    call the model ONCE, normalize, and save to resource_configs/.
    """
    st.divider()
    st.subheader("Resource Configuration (full-context, single call)")

    if not lab_name:
        st.info("Select a lab first.")
        return

    content_key = f"content_docs/{sanitize_lab_name(lab_name)}.txt"
    if not s3_object_exists(s3, bucket, content_key):
        st.error(f"Content file not found for lab: {lab_name}")
        return

    logs_key = (st.session_state.get("selected_lab_files", {}) or {}).get("execution_logs_key")
    if not logs_key:
        logs_key = _pick_latest_logs_key(bucket, s3, lab_name)

    include_logs= False
    max_chars= False

    # include_logs = st.checkbox("Include execution logs to enrich extraction (recommended)", value=bool(logs_key))

    # with st.expander("Advanced: context limits", expanded=False):
    #     max_chars = st.number_input("Max context characters (0 = full file)", min_value=0, value=0, step=100000,
    #                                 help="Set to 0 to pass the entire content file.")

    question = "Extract all AWS resources the learner creates/configures in this lab and their configurations as JSON."
    run = st.button("Generate & Save Resource Config JSON", type="primary", key=f"gen_rc_{lab_name}")

    existing_rc_key = f"resource_configs/{sanitize_lab_name(lab_name)}_resource_config.json"
    if s3_object_exists(s3, bucket, existing_rc_key):
        with st.expander("Existing resource config for this lab"):
            view_s3_file_inline(s3, bucket, existing_rc_key)
            st.caption(f"s3://{bucket}/{existing_rc_key}")

    if not run:
        return

    client = _openai_client()
    if not client:
        return

    with st.spinner("Assembling full context…"):
        content_text = _load_text_from_s3(s3, bucket, content_key) or ""
        if not content_text.strip():
            st.error("Content file is empty.")
            return

        if include_logs and logs_key:
            logs_bytes = s3_get_bytes(s3, bucket, logs_key)
            if logs_bytes:
                try:
                    logs_text = json.dumps(json.loads(logs_bytes.decode("utf-8", errors="replace")))[:8000]
                except Exception:
                    logs_text = logs_bytes.decode("utf-8", errors="replace")[:8000]
                content_text += "\n\nExecution logs (truncated):\n" + logs_text

        if max_chars:
            context = content_text[:max_chars]
        else:
            context = content_text

    with st.spinner("Extracting resources (single model call)…"):
        raw_payload = call_openai_extract(client, context, question)
        normalized = normalize_payload(raw_payload)

    rc_key = f"resource_configs/{sanitize_lab_name(lab_name)}_resource_config.json"
    try:
        s3_put_json(s3, bucket, rc_key, normalized)
        st.success(f"Saved resource config to s3://{bucket}/{rc_key}")
        view_s3_file_inline(s3, bucket, rc_key)
        st.session_state.setdefault("selected_lab_files", {})
        st.session_state["selected_lab_files"]["resource_config_key"] = rc_key
    except Exception as e:
        st.error(f"Failed to write resource config: {e}")

# -----------------------------
# Main Lab Extraction Module  — streamlined auto-flow
# -----------------------------
def _init_flow_state():
    """Initialise all session-state keys used by the auto-flow exactly once."""
    defaults = {
        # inputs
        "flow_lab_url": "",
        "flow_user_id": "",
        "flow_iam_username": "",
        # derived
        "flow_editor_url": "",
        "flow_lab_name": "",          # sanitized name used as S3 key
        "flow_lab_display_name": "",  # human-readable name from [Task 1]
        # progress flags
        "flow_lab_extracted": False,
        "flow_history_loaded": False,
        "flow_logs_fetched": False,
        "flow_rc_generated": False,
        # data
        "flow_attempts": [],
        "flow_selected_attempt": None,
        "flow_selected_region": "us-east-1",
        "flow_execution_log_key": None,
        "flow_resource_config_key": None,
        # analysis reports
        "flow_error_analysis_key": None,
        "flow_compliance_key": None,
        "flow_error_analysis_done": False,
        "flow_compliance_done": False,
        # user-chosen options (set before flow starts)
        "flow_opt_force_extract": False,    # re-extract latest lab content + RC
        "flow_opt_fetch_validation": False, # fetch attempt validation results
        # validation data
        "flow_validation_data": None,       # raw validation API response dict
        "flow_validation_fetched": False,
        # whether the flow has been started at all (button guard)
        "flow_started": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _auto_extract_lab(*, s3, bucket: str, root_url: str, cookie_header: str, status_container) -> bool:
    """
    Run Playwright extraction for the lab at root_url, save to S3, auto-detect lab name.
    Returns True on success.
    """
    try:
        ensure_chromium_installed()
        try:
            from playwright.sync_api import sync_playwright
        except Exception:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            from playwright.sync_api import sync_playwright

        links: List[TaskLink] = []
        arts: List[TaskArtifact] = []
        landing_artifact: Optional[TaskArtifact] = None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            ctx = browser.new_context(user_agent=USER_AGENT, ignore_https_errors=True)
            cookie_value = ""
            if cookie_header:
                line = cookie_header
                if ":" in line:
                    k2, v2 = line.split(":", 1)
                    if k2.strip().lower() == "cookie":
                        cookie_value = v2.strip()
                else:
                    cookie_value = line.strip()
            if cookie_value:
                ctx.add_cookies(parse_cookie_header_to_playwright_cookies(cookie_value, root_url))

            page = ctx.new_page()
            page.set_default_timeout(20000)

            page.goto(root_url, wait_until="domcontentloaded")
            try:
                page.wait_for_load_state("networkidle")
            except Exception:
                pass

            landing_slate = pw_get_slate_text(page)
            ace_vals = pw_get_all_ace_values(page)
            landing_blocks = [{"type": "ace", "content": v} for v in ace_vals if str(v).strip()]
            landing_title = page.title() or root_url
            landing_artifact = TaskArtifact(url=root_url, title=landing_title, slate_text=landing_slate, code_blocks=landing_blocks)

            anchors = page.eval_on_selector_all(
                'a[href*="cloudlabeditor"]',
                '''els => els.map(el => {
                    function categoryOf(node){
                      let p = node;
                      for (let i=0;i<20 && p;i++){
                        const s = p.querySelector && p.querySelector('span.text-xl');
                        if (s && s.textContent.trim()) return s.textContent.trim();
                        p = p.parentElement;
                      }
                      return null;
                    }
                    return {
                      href: el.getAttribute('href') || '',
                      title: (el.textContent || '').trim(),
                      category: categoryOf(el)
                    };
                })'''
            )
            for a in anchors:
                href_abs = normalize_url(root_url, a.get("href", ""))
                if href_abs:
                    links.append(TaskLink(category=a.get("category"), title=a.get("title", ""), href=href_abs))

            if not links:
                html = page.content() or ""
                for href in re.findall(r"/(?:editor/)?cloudlabeditor/[0-9]+/[0-9]+/[0-9]+/[0-9]+", html):
                    href_abs = normalize_url(root_url, href)
                    if href_abs:
                        links.append(TaskLink(category=None, title="", href=href_abs))

            for tl in links:
                try:
                    page.goto(tl.href, wait_until="domcontentloaded")
                    try:
                        page.wait_for_load_state("networkidle")
                    except Exception:
                        pass
                    slate_text = pw_get_slate_text(page)
                    per_task_code_vals = pw_get_all_ace_values(page)
                    code_blocks = [{"type": "ace", "content": v} for v in per_task_code_vals if str(v).strip()]
                    title = tl.title or (page.title() or tl.href)
                    arts.append(TaskArtifact(url=tl.href, title=title, slate_text=slate_text, code_blocks=code_blocks))
                    time.sleep(0.2)
                except Exception:
                    arts.append(TaskArtifact(url=tl.href, title=tl.title or tl.href, slate_text="", code_blocks=[]))

            ctx.close()
            browser.close()

        status_container.write(f"Extracted {len(links)} task(s).")

        # Use a temporary sanitized name derived from URL IDs so we can save first
        ids = parse_editor_ids(root_url)
        temp_name = f"lab_{'_'.join(ids)}" if ids else sanitize_lab_name(f"lab_{hashlib.sha256(root_url.encode()).hexdigest()[:8]}")

        content_key, policy_key, prescript_key = s3_write_outputs_to_folders(
            s3=s3, bucket=bucket, lab_name=temp_name,
            root_url=root_url, landing=landing_artifact, tasks=arts,
        )

        # Auto-detect display name from [Task 1] line
        content_bytes = s3_get_bytes(s3, bucket, content_key)
        display_name = ""
        if content_bytes:
            content_text = content_bytes.decode("utf-8", errors="replace")
            display_name = extract_lab_display_name_from_content(content_text) or ""

        # If we got a real display name, rename files to use it
        if display_name:
            rename_lab_files_in_s3(s3, bucket, temp_name, display_name)
            final_name = sanitize_lab_name(display_name)
        else:
            final_name = temp_name
            display_name = temp_name

        st.session_state["flow_lab_name"] = final_name
        st.session_state["flow_lab_display_name"] = display_name
        st.session_state["flow_lab_extracted"] = True

        # ── Register in DynamoDB so future runs skip extraction ───────────────
        lab_id = _lab_id_from_url(root_url)
        if lab_id:
            dynamo_put_lab(
                lab_id=lab_id,
                lab_display_name=display_name,
                lab_s3_name=final_name,
                root_url=root_url,
            )

        # Load lab content + policy + prescript into session for downstream use
        sel = {
            "content_key": f"content_docs/{final_name}.txt",
            "policy_key": f"policies/{final_name}.json",
            "prescript_key": f"prescript/{final_name}.json",
        }
        hydrate_selected_lab_assets_into_session(bucket, s3, sel)

        return True
    except Exception as e:
        st.error(f"Lab extraction failed: {e}")
        return False


def _auto_fetch_history(*, editor_url: str, user_id: str, cookie_header: str) -> List[Dict[str, Any]]:
    history_url = build_history_api_url(editor_url, user_id)
    if not history_url:
        return []
    try:
        history_json = fetch_attempt_history(history_url, cookie_header)
        return parse_attempts_from_history(history_json)
    except Exception as e:
        st.warning(f"Could not fetch attempts history: {e}")
        return []


def _auto_fetch_logs(*, s3, bucket: str, editor_url: str, user_id: str, attempt: Dict[str, Any],
                     lab_name: str, cookie_header: str, iam_username: str = "") -> Optional[str]:
    """
    Fetch execution logs for the given attempt and save to S3. Returns the S3 key.

    Key scheme:
      execution_logs/<lab>_<user_id>_<attempt_id>.json          (no IAM user)
      execution_logs/<lab>_<user_id>_<attempt_id>_iam_<iam>.json (IAM user specified)

    If the key already exists in S3 the API call is skipped and the existing
    file is loaded directly — unless a re-fetch is explicitly requested.
    """
    attempt_id = attempt.get("attempt_id", "")
    region     = attempt.get("region") or "us-east-1"

    # Build the S3 key — include IAM username when provided so each variant is unique
    iam_suffix = f"_iam_{iam_username.strip()}" if (iam_username or "").strip() else ""
    key = f"execution_logs/{sanitize_lab_name(lab_name)}_{user_id}_{attempt_id}{iam_suffix}.json"

    # ── Return cached file if it already exists ───────────────────────────────
    if s3_object_exists(s3, bucket, key):
        existing = s3_get_bytes(s3, bucket, key)
        if existing:
            st.session_state["log_content"] = existing.decode("utf-8", errors="replace")
            st.session_state.setdefault("selected_lab_files", {})["execution_logs_key"] = key
            return key  # reuse without API call

    # ── Fetch from API ────────────────────────────────────────────────────────
    api_url = build_events_api_url(editor_url, user_id, attempt_id, region, iam_username=iam_username)
    if not api_url:
        st.warning("Could not build events API URL.")
        return None
    try:
        logs = fetch_execution_logs(api_url, cookie_header)
        s3_put_json(s3, bucket, key, logs)
        st.session_state["log_content"] = json.dumps(logs)
        st.session_state.setdefault("selected_lab_files", {})["execution_logs_key"] = key
        return key
    except Exception as e:
        st.warning(f"Failed to fetch execution logs: {e}")
        return None


def _auto_generate_resource_config(*, s3, bucket: str, lab_name: str,
                                    force: bool = False) -> Optional[str]:
    """
    Generate resource config JSON and save to S3. Returns the S3 key.
    If the file already exists in S3 and force=False, loads it directly
    without calling the AI model again.
    """
    rc_key = f"resource_configs/{sanitize_lab_name(lab_name)}_resource_config.json"

    # ── Check if already exists in S3 ────────────────────────────────────────
    if not force and s3_object_exists(s3, bucket, rc_key):
        existing = s3_get_bytes(s3, bucket, rc_key)
        if existing:
            st.session_state.setdefault("selected_lab_files", {})["resource_config_key"] = rc_key
            st.session_state["cloudtrail_lab_spec_content"] = existing.decode("utf-8", errors="replace")
            return rc_key  # reuse without re-generating

    # ── Generate from content file ────────────────────────────────────────────
    content_key = f"content_docs/{sanitize_lab_name(lab_name)}.txt"
    if not s3_object_exists(s3, bucket, content_key):
        return None
    client = _openai_client()
    if not client:
        return None
    content_text = _load_text_from_s3(s3, bucket, content_key) or ""
    if not content_text.strip():
        return None
    question = "Extract all AWS resources the learner creates/configures in this lab and their configurations as JSON."
    raw_payload = call_openai_extract(client, content_text, question)
    normalized = normalize_payload(raw_payload)
    try:
        s3_put_json(s3, bucket, rc_key, normalized)
        st.session_state.setdefault("selected_lab_files", {})["resource_config_key"] = rc_key
        st.session_state["cloudtrail_lab_spec_content"] = json.dumps(normalized, indent=2)
        return rc_key
    except Exception as e:
        st.warning(f"Failed to save resource config: {e}")
        return None


def build_validation_api_url(editor_url: str, user_id: str, attempt_id: str) -> Optional[str]:
    """
    Build the attempt-validation API URL.
    Pattern: https://www.educative.io/api/cloud-lab/attempts/{id1}/{id2}/{id3}/{user_id}?attempt_id={attempt_id}
    """
    ids = parse_editor_ids(editor_url)
    if not ids:
        return None
    id1, id2, id3 = ids
    if not (user_id and attempt_id):
        return None
    return (
        f"https://www.educative.io/api/cloud-lab/attempts/{id1}/{id2}/{id3}"
        f"/{user_id.strip()}?attempt_id={attempt_id}"
    )


def fetch_validation_results(api_url: str, cookie_header: str) -> Optional[Dict[str, Any]]:
    """Call the validation API and return the parsed JSON response."""
    import requests as _requests
    headers = {}
    if cookie_header:
        ch = cookie_header.strip()
        if not ch.lower().startswith("cookie"):
            ch = "Cookie: " + ch
        k, v = ch.split(":", 1)
        headers[k.strip()] = v.strip()
    headers["User-Agent"] = USER_AGENT
    resp = _requests.get(api_url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _display_validation_results(data: Dict[str, Any]):
    """Render the validation API response in a structured, readable way."""
    attempt = data.get("attempt", {})
    if not attempt:
        st.warning("No attempt data in validation response.")
        return

    # ── Header metrics ────────────────────────────────────────────────────────
    progress = attempt.get("progress", {})
    env      = attempt.get("environment", {})
    status   = attempt.get("status", {})

    st.markdown("#### Attempt Overview")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Attempt #",     attempt.get("id", "—"))
    m2.metric("Completed",     progress.get("completed", 0))
    m3.metric("In Progress",   progress.get("in_progress", 0))
    m4.metric("Not Started",   progress.get("not_started", 0))
    pct = progress.get("completion_percentage", 0)
    m5.metric("Completion",    f"{pct:.1f}%")

    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"**Setup:** {status.get('setup', '—')}")
        st.caption(f"**Termination:** {status.get('termination', '—')}")
    with col_b:
        st.caption(f"**AWS Account:** {env.get('account_id', '—')}")
        st.caption(f"**Start:** {attempt.get('start_time', '—')}")
        st.caption(f"**End:** {attempt.get('end_time', '—')}")

    st.divider()

    # ── Tasks table ───────────────────────────────────────────────────────────
    tasks = attempt.get("tasks", [])
    if not tasks:
        st.info("No task data available.")
        return

    st.markdown("#### Task Results")

    # Build numbered rows — task number derived from list position (1-indexed)
    rows = []
    for i, t in enumerate(tasks, start=1):
        task_status = t.get("status", "unknown")
        icon = {
            "completed":   "✅",
            "in_progress": "🔄",
            "not_started": "⬜",
            "tried":       "🟡",
        }.get(task_status, "❓")

        # Combine message + failure_message into a single cell
        msg = t.get("message", "—") or "—"
        failure = (t.get("failure_message") or "").strip()
        combined_message = f"{msg} — {failure}" if failure else msg

        rows.append({
            "#":             i,
            "Task ID":       t.get("task_id", "—"),
            "Status":        f"{icon} {task_status}",
            "Message":       combined_message,
            "Score":         t.get("score", 0),
            "Total Tests":   t.get("total_tests", 0),
            "Passed Tests":  t.get("passed_tests", 0),
            "Attempts":      t.get("attempts", 0),
            "Last Attempt":  t.get("last_attempt") or "—",
        })

    st.dataframe(rows, use_container_width=True)

    # Highlight tasks that have notable messages (completed tasks excluded from noise)
    generic = {"Task not yet started", "Task completed", "—"}
    notable = [r for r in rows if r["Message"] not in generic]
    if notable:
        st.markdown("**Notable task messages:**")
        for r in notable:
            if "completed" in r["Status"]:
                colour = "green"
            elif "tried" in r["Status"]:
                colour = "orange"
            else:
                colour = "gray"
            st.markdown(f"- Task **#{r['#']}** (ID `{r['Task ID']}`): :{colour}[{r['Message']}]")


def _run_cached_analyses(*, s3, bucket: str, lab_name: str, user_id: str, attempt_id: str,
                         iam_username: str = ""):
    """
    Auto-run error analysis and resource compliance if not already done for this
    lab+user+attempt combination.  Results are saved to S3 so revisiting is instant.
    Both analyses require organized logs AND a resource config spec to be in session.
    """
    organized_logs = st.session_state.get("cloudtrail_organized_logs")
    lab_spec       = st.session_state.get("cloudtrail_lab_spec_content")

    if not organized_logs or not lab_spec:
        # Prerequisites not met — analysis tabs will show an info message
        return

    error_key      = _report_key(lab_name, user_id, attempt_id, "error_analysis", iam_username)
    compliance_key = _report_key(lab_name, user_id, attempt_id, "compliance",     iam_username)

    # ── Step 6: Error analysis ────────────────────────────────────────────────
    if not st.session_state.get("flow_error_analysis_done"):
        # Try to load a previously saved report first
        saved = load_report_from_s3(s3, bucket, error_key)
        if saved:
            st.session_state["gemini_lab_analysis"]      = saved
            st.session_state["flow_error_analysis_key"]  = error_key
            st.session_state["flow_error_analysis_done"] = True
        else:
            with st.status("⚙️ Step 6 — Analyzing errors in lab…", expanded=True) as status:
                try:
                    asyncio.run(analyze_lab_with_gemini(organized_logs, lab_spec))
                    report = st.session_state.get("gemini_lab_analysis", "")
                    if report:
                        save_report_to_s3(s3, bucket, error_key, report)
                        st.session_state["flow_error_analysis_key"]  = error_key
                        st.session_state["flow_error_analysis_done"] = True
                        status.update(label="✅ Error analysis complete", state="complete")
                    else:
                        status.update(label="⚠️ Error analysis returned empty result", state="error")
                except Exception as e:
                    status.update(label=f"⚠️ Error analysis failed: {e}", state="error")

    # ── Step 7: Resource compliance ───────────────────────────────────────────
    if not st.session_state.get("flow_compliance_done"):
        saved = load_report_from_s3(s3, bucket, compliance_key)
        if saved:
            st.session_state["gemini_resource_analysis"] = saved
            st.session_state["flow_compliance_key"]      = compliance_key
            st.session_state["flow_compliance_done"]     = True
        else:
            with st.status("⚙️ Step 7 — Checking resource compliance…", expanded=True) as status:
                try:
                    asyncio.run(analyze_resource_creation(organized_logs, lab_spec))
                    report = st.session_state.get("gemini_resource_analysis", "")
                    if report:
                        save_report_to_s3(s3, bucket, compliance_key, report)
                        st.session_state["flow_compliance_key"]  = compliance_key
                        st.session_state["flow_compliance_done"] = True
                        status.update(label="✅ Resource compliance complete", state="complete")
                    else:
                        status.update(label="⚠️ Compliance check returned empty result", state="error")
                except Exception as e:
                    status.update(label=f"⚠️ Compliance check failed: {e}", state="error")


def lab_extractor_module():
    st.header("Lab Content Extractor")
    _init_flow_state()

    # --- S3 init
    bucket, s3 = get_s3()
    if not bucket or not s3:
        st.error("Missing S3 setup. Add [aws] bucket/region/credentials to secrets.toml.")
        st.stop()

    # --- Cookie
    default_cookie = (st.secrets.get("cookies", {}) or {}).get("educative", "")
    cookie_source = st.selectbox(
        "Cookie source",
        options=("Use cookie from secrets", "Enter custom cookie"),
        index=0 if default_cookie else 1,
    )
    if cookie_source == "Use cookie from secrets":
        cookie_preview = (default_cookie[:80] + "…") if len(default_cookie) > 80 else default_cookie
        st.text_input("Cookie (from secrets, read-only)", value=cookie_preview, disabled=True)
        cookie_header = default_cookie.strip()
    else:
        raw_custom = st.text_area("Custom Cookie (paste full `Cookie:` line or just the cookie value)", height=80)
        cookie_header = raw_custom.strip()
    if cookie_header and not cookie_header.lower().startswith("cookie"):
        cookie_header = "Cookie: " + cookie_header

    st.divider()

    # ── SECONDARY PATH: Use existing lab from S3 ──────────────────────────────
    with st.expander("📂 Use an existing lab from S3 (optional)", expanded=False):
        with st.spinner("Listing labs in S3…"):
            labs = s3_list_labs(bucket, s3)
        if labs:
            lab_choice = st.selectbox("Choose a saved lab", options=labs, index=0, key="existing_lab_choice")
            if lab_choice and st.button("Load existing lab", key="load_existing_lab"):
                sel = {
                    "content_key": f"content_docs/{lab_choice}.txt",
                    "policy_key": f"policies/{lab_choice}.json",
                    "prescript_key": f"prescript/{lab_choice}.json",
                }
                hydrate_selected_lab_assets_into_session(bucket, s3, sel)
                st.session_state["flow_lab_name"] = lab_choice
                st.session_state["flow_lab_display_name"] = lab_choice
                st.session_state["flow_lab_extracted"] = True
                st.session_state["flow_lab_url"] = get_root_url_from_content_file(s3, bucket, lab_choice) or ""
                st.session_state["flow_editor_url"] = normalize_lab_url_to_editor(st.session_state["flow_lab_url"])
                st.session_state["flow_history_loaded"] = False
                st.session_state["flow_logs_fetched"] = False
                st.session_state["flow_rc_generated"] = s3_object_exists(s3, bucket, f"resource_configs/{lab_choice}_resource_config.json")
                if st.session_state["flow_rc_generated"]:
                    rc_bytes = s3_get_bytes(s3, bucket, f"resource_configs/{lab_choice}_resource_config.json")
                    if rc_bytes:
                        st.session_state["cloudtrail_lab_spec_content"] = rc_bytes.decode("utf-8", errors="replace")
                st.success(f"Loaded lab: {lab_choice}")
        else:
            st.info("No labs found in S3 yet.")

    # ── S3 Browser ────────────────────────────────────────────────────────────
    with st.expander("🔎 Browse S3 files", expanded=False):
        s3_prefix_browser_ui(bucket, s3)

    st.divider()

    # ── PRIMARY INPUTS: URL + User ID + IAM username ──────────────────────────
    st.subheader("Analyze a Lab Session")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        lab_url_input = st.text_input(
            "Lab URL (published or editor link)",
            placeholder="https://www.educative.io/collection/page/.../cloudlab  or  .../editor/cloudlabeditor/...",
            value=st.session_state["flow_lab_url"],
            key="input_lab_url",
        )
    with col2:
        user_id_input = st.text_input(
            "Learner ID",
            value=st.session_state.get("flow_user_id", ""),
            placeholder="user_123",
            key="input_user_id",
        )
    with col3:
        iam_username_input = st.text_input(
            "IAM Username (optional)",
            value=st.session_state.get("flow_iam_username", ""),
            placeholder="e.g. lab-user-abc",
            help="Leave blank if the lab does not use a dedicated IAM user. "
                 "When provided, appends &iam_username=... to the CloudTrail log API call.",
            key="input_iam_username",
        )

    # ── Options checkboxes ────────────────────────────────────────────────────
    st.markdown("**Options**")
    opt_col1, opt_col2 = st.columns(2)
    with opt_col1:
        opt_force_extract = st.checkbox(
            "🔄 Extract latest content for the lab",
            value=st.session_state.get("flow_opt_force_extract", False),
            key="opt_force_extract",
            help="Re-runs lab extraction and regenerates the resource configuration file, "
                 "even if they already exist in S3.",
        )
    with opt_col2:
        opt_fetch_validation = st.checkbox(
            "✅ Fetch attempt validation results",
            value=st.session_state.get("flow_opt_fetch_validation", False),
            key="opt_fetch_validation",
            help="Calls the validation API for the selected attempt and shows per-task scores and messages.",
        )

    # ── Detect input changes → reset flow guard ───────────────────────────────
    inputs_changed = (
        lab_url_input         != st.session_state["flow_lab_url"]
        or user_id_input      != st.session_state.get("flow_user_id", "")
        or iam_username_input != st.session_state.get("flow_iam_username", "")
    )
    if inputs_changed:
        prev_iam = st.session_state.get("flow_iam_username", "")
        st.session_state["flow_iam_username"] = iam_username_input
        st.session_state["flow_user_id"]      = user_id_input
        st.session_state["flow_lab_url"]      = lab_url_input
        editor_url = normalize_lab_url_to_editor(lab_url_input)
        st.session_state["flow_editor_url"]   = editor_url
        if lab_url_input != st.session_state.get("_prev_lab_url", ""):
            for k in ("flow_lab_extracted", "flow_history_loaded", "flow_logs_fetched",
                      "flow_rc_generated", "flow_error_analysis_done", "flow_compliance_done",
                      "flow_validation_fetched", "flow_started"):
                st.session_state[k] = False
            st.session_state["flow_attempts"]         = []
            st.session_state["flow_selected_attempt"] = None
            st.session_state["flow_validation_data"]  = None
        elif iam_username_input != prev_iam:
            for k in ("flow_logs_fetched", "flow_error_analysis_done",
                      "flow_compliance_done", "flow_validation_fetched"):
                st.session_state[k] = False
            st.session_state["flow_validation_data"] = None
        st.session_state["_prev_lab_url"] = lab_url_input

    # Sync option flags
    st.session_state["flow_opt_force_extract"]    = opt_force_extract
    st.session_state["flow_opt_fetch_validation"] = opt_fetch_validation

    # ── Start Analysis button ─────────────────────────────────────────────────
    can_start = bool(lab_url_input.strip() and user_id_input.strip())
    if st.button(
        "▶ Start Analysis",
        type="primary",
        disabled=not can_start,
        key="btn_start_flow",
        help="Fill in Lab URL and Learner ID to enable.",
    ):
        for k in ("flow_lab_extracted", "flow_history_loaded", "flow_logs_fetched",
                  "flow_rc_generated", "flow_error_analysis_done", "flow_compliance_done",
                  "flow_validation_fetched"):
            st.session_state[k] = False
        st.session_state["flow_attempts"]         = []
        st.session_state["flow_selected_attempt"] = None
        st.session_state["flow_validation_data"]  = None
        st.session_state["flow_started"]          = True
        st.rerun()

    if not st.session_state.get("flow_started"):
        st.info("Fill in the fields above and click **▶ Start Analysis** to begin.")
        return

    # ── Flow is now running ───────────────────────────────────────────────────
    lab_url      = st.session_state["flow_lab_url"].strip()
    user_id      = st.session_state["flow_user_id"].strip()
    iam_username = st.session_state.get("flow_iam_username", "").strip()
    editor_url   = st.session_state["flow_editor_url"]

    if not lab_url or not user_id:
        st.info("Enter a Lab URL and Learner ID above to begin.")
        return

    ids = parse_editor_ids(lab_url)
    if not ids:
        st.error("Could not extract lab IDs from URL. Please check the URL format.")
        return

    # ── STEP 1: Extract lab — respect flow_opt_force_extract option ──────────
    force_extract = st.session_state.get("flow_opt_force_extract", False)
    if not st.session_state["flow_lab_extracted"]:
        lab_id = _lab_id_from_url(lab_url)
        cached = dynamo_get_lab(lab_id) if lab_id else None

        if cached and not force_extract:
            # Lab exists in DynamoDB and user did NOT request fresh extraction
            st.info(
                f"📋 Lab **{cached.get('lab_display_name', cached.get('lab_s3_name', lab_id))}** "
                f"was last extracted on `{cached.get('extracted_at', 'unknown')}`. Using cached version."
            )
            final_name    = cached["lab_s3_name"]
            display_name  = cached.get("lab_display_name", final_name)
            cached_editor = cached.get("root_url", editor_url)
            st.session_state["flow_lab_name"]          = final_name
            st.session_state["flow_lab_display_name"]  = display_name
            st.session_state["flow_editor_url"]        = cached_editor
            st.session_state["flow_lab_extracted"]     = True
            st.session_state["_flow_force_reextract"]  = False
            sel = {
                "content_key":   f"content_docs/{final_name}.txt",
                "policy_key":    f"policies/{final_name}.json",
                "prescript_key": f"prescript/{final_name}.json",
            }
            hydrate_selected_lab_assets_into_session(bucket, s3, sel)
            # Load resource config if already generated (no regeneration needed)
            rc_key = f"resource_configs/{final_name}_resource_config.json"
            if s3_object_exists(s3, bucket, rc_key):
                rc_bytes = s3_get_bytes(s3, bucket, rc_key)
                if rc_bytes:
                    st.session_state["cloudtrail_lab_spec_content"] = rc_bytes.decode("utf-8", errors="replace")
                    st.session_state["flow_rc_generated"] = True
            st.success(f"✅ Loaded from cache: **{display_name}**")
        else:
            # Either first time OR user checked "Extract latest content"
            label = "⚙️ Step 1 — Re-extracting lab content…" if (cached and force_extract) else "⚙️ Step 1 — Extracting lab content…"
            st.session_state["_flow_force_reextract"] = force_extract
            if force_extract:
                st.session_state["flow_rc_generated"] = False  # force RC regeneration too
            with st.status(label, expanded=True) as status:
                ok = _auto_extract_lab(s3=s3, bucket=bucket, root_url=editor_url,
                                       cookie_header=cookie_header, status_container=status)
                if ok:
                    status.update(label=f"✅ Lab extracted: {st.session_state['flow_lab_display_name']}", state="complete")
                else:
                    status.update(label="❌ Lab extraction failed", state="error")
                    return

    lab_name = st.session_state["flow_lab_name"]
    display_name = st.session_state["flow_lab_display_name"]
    st.success(f"📚 Lab: **{display_name}**")

    # ── STEP 2: Fetch attempt history ─────────────────────────────────────────
    if not st.session_state["flow_history_loaded"]:
        with st.status("⚙️ Step 2 — Fetching attempt history…", expanded=True) as status:
            attempts = _auto_fetch_history(editor_url=editor_url, user_id=user_id, cookie_header=cookie_header)
            st.session_state["flow_attempts"] = attempts
            st.session_state["flow_history_loaded"] = True
            status.update(label=f"✅ Found {len(attempts)} attempt(s)", state="complete")

    attempts = st.session_state["flow_attempts"]
    if not attempts:
        st.warning("No attempts found for this learner on this lab.")
        return

    # ── STEP 3: Let user pick an attempt ──────────────────────────────────────
    st.subheader("Select Attempt to Analyze")
    attempt_rows = []
    for a in attempts:
        row = {"attempt_id": a.get("attempt_id", ""), "started": a.get("started", ""), "ended": a.get("ended", "")}
        if a.get("meta"):
            row.update(a["meta"])
        attempt_rows.append(row)
    st.dataframe(attempt_rows, use_container_width=True)

    attempt_ids = [a.get("attempt_id", "") for a in attempts if a.get("attempt_id")]
    if not attempt_ids:
        st.warning("No attempt IDs found in history response.")
        return

    # Default to last (most recent) attempt
    default_idx = len(attempt_ids) - 1
    current_sel = st.session_state.get("flow_selected_attempt")
    current_idx = attempt_ids.index(current_sel) if (current_sel in attempt_ids) else default_idx

    chosen_attempt_id = st.selectbox("Choose attempt to analyze", options=attempt_ids,
                                     index=current_idx, key="attempt_selector")

    # When selection changes, reset only attempt-specific flags (logs + analyses).
    # RC is per-lab, not per-attempt — never reset it here.
    if chosen_attempt_id != st.session_state.get("flow_selected_attempt"):
        st.session_state["flow_selected_attempt"] = chosen_attempt_id
        st.session_state["flow_logs_fetched"]          = False
        st.session_state["flow_error_analysis_done"]   = False
        st.session_state["flow_compliance_done"]       = False
        st.session_state["flow_error_analysis_key"]    = None
        st.session_state["flow_compliance_key"]        = None
        st.session_state["flow_validation_fetched"]    = False
        st.session_state["flow_validation_data"]       = None

    chosen_attempt = next((a for a in attempts if a.get("attempt_id") == chosen_attempt_id), {})

    # ── STEP 4: Fetch execution logs (cached in S3 if already present) ────────
    if not st.session_state["flow_logs_fetched"]:
        with st.status("⚙️ Step 4 — Loading execution logs…", expanded=True) as status:
            log_key = _auto_fetch_logs(
                s3=s3, bucket=bucket, editor_url=editor_url, user_id=user_id,
                attempt=chosen_attempt, lab_name=lab_name, cookie_header=cookie_header,
                iam_username=iam_username,
            )
            if log_key:
                st.session_state["flow_execution_log_key"] = log_key
                st.session_state["flow_logs_fetched"] = True
                # Also initialise cloudtrail organised logs
                raw_logs = s3_get_bytes(s3, bucket, log_key)
                if raw_logs:
                    class _MockFile:
                        def __init__(self, data): self.data = data; self.pos = 0
                        def read(self, n=None):
                            if n is None: chunk = self.data[self.pos:]; self.pos = len(self.data); return chunk
                            chunk = self.data[self.pos:self.pos+n]; self.pos += n; return chunk
                        def seek(self, p): self.pos = p
                    st.session_state["cloudtrail_organized_logs"] = load_and_process_logs(_MockFile(raw_logs))
                    st.session_state["cloudtrail_selected_log_file"] = log_key
                status.update(label="✅ Execution logs fetched", state="complete")
            else:
                status.update(label="⚠️ Could not fetch execution logs", state="error")

    # ── STEP 5: Resource config (skip if already in S3, force only on re-extract) ──
    if not st.session_state["flow_rc_generated"]:
        force_rc = st.session_state.get("_flow_force_reextract", False)
        with st.status(
            "⚙️ Step 5 — Loading resource configuration…" if not force_rc
            else "⚙️ Step 5 — Re-generating resource configuration…",
            expanded=True
        ) as status:
            rc_key = _auto_generate_resource_config(s3=s3, bucket=bucket, lab_name=lab_name, force=force_rc)
            if rc_key:
                st.session_state["flow_resource_config_key"] = rc_key
                st.session_state["flow_rc_generated"] = True
                was_cached = not force_rc and s3_object_exists(s3, bucket, rc_key)
                status.update(
                    label="✅ Resource config loaded from S3" if was_cached else "✅ Resource config generated",
                    state="complete"
                )
            else:
                status.update(label="⚠️ Resource config generation failed (check OpenAI key)", state="error")

    # ── STEP 6 & 7: Auto-run error analysis + compliance (with S3 caching) ───
    _run_cached_analyses(s3=s3, bucket=bucket, lab_name=lab_name,
                         user_id=user_id, attempt_id=chosen_attempt_id,
                         iam_username=iam_username)

    # ── Summary of loaded data ────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Session Summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.session_state.get("lab_content"):
            st.success("✅ Lab content loaded")
        else:
            st.warning("⚠️ Lab content not loaded")
    with c2:
        if st.session_state.get("flow_logs_fetched"):
            st.success(f"✅ Logs: `{st.session_state.get('flow_execution_log_key', '').split('/')[-1]}`")
        else:
            st.warning("⚠️ Logs not loaded")
    with c3:
        if st.session_state.get("flow_rc_generated"):
            st.success("✅ Resource config generated")
        else:
            st.warning("⚠️ Resource config not generated")

    # ── STEP 8: Attempt validation results (only if option is checked) ────────
    if st.session_state.get("flow_opt_fetch_validation"):
        st.divider()
        st.subheader("🔎 Attempt Validation Results")

        if not st.session_state.get("flow_validation_fetched"):
            validation_url = build_validation_api_url(editor_url, user_id, chosen_attempt_id)
            if not validation_url:
                st.warning("Could not build validation API URL — check the lab URL format.")
            else:
                with st.status("⚙️ Step 8 — Fetching validation results…", expanded=True) as vstatus:
                    try:
                        vdata = fetch_validation_results(validation_url, cookie_header)
                        st.session_state["flow_validation_data"]    = vdata
                        st.session_state["flow_validation_fetched"] = True
                        vstatus.update(label="✅ Validation results fetched", state="complete")
                    except Exception as e:
                        vstatus.update(label=f"⚠️ Validation fetch failed: {e}", state="error")

        if st.session_state.get("flow_validation_fetched") and st.session_state.get("flow_validation_data"):
            _display_validation_results(st.session_state["flow_validation_data"])
            if st.button("🔄 Re-fetch validation results", key="refetch_validation_btn"):
                st.session_state["flow_validation_fetched"] = False
                st.session_state["flow_validation_data"]    = None
                st.rerun()


import chromadb
from chromadb.config import Settings as ChromaClientSettings
import uuid
from typing import Optional, Any
# at top-level imports
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
CHROMA_DB_DIR = "./chroma_db"
RESOURCES_DIR = "./resources"
FEEDBACK_RESPONSES_DIR = "./feedback_responses"
LOGS_DIR = "./logs"

# Ensure directories exist
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(RESOURCES_DIR, exist_ok=True)
os.makedirs(FEEDBACK_RESPONSES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
def _chroma_client():
    # Optional settings; telemetry off to reduce noise
    return chromadb.PersistentClient(
        path=CHROMA_DB_DIR,
        settings=ChromaClientSettings(anonymized_telemetry=False)
    )

def _resolve_embedding_or_raise() -> Tuple[Any, int]:
    """
    Return a working embedding model and its dimension.
    Raises a clear RuntimeError if not available, instead of letting None propagate.
    """
    # Prefer OpenAI
    api_key = (st.secrets.get("openaikey", {}).get("key"))

    emb = None
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        from llama_index.embeddings.openai import OpenAIEmbedding
        emb = OpenAIEmbedding(model="text-embedding-3-small")
    else:
        # Fallback: local HF
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        emb = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # --- sanity check (prevents None vectors from leaking)
    vec = emb.get_text_embedding("ok")
    if not vec or not all(isinstance(x, (int, float)) for x in vec):
        raise RuntimeError("Embedding backend returned invalid vector.")
    dim = len(vec)

    # set global default for LlamaIndex
    Settings.embed_model = emb
    return emb, dim


def _resolve_embedding_or_none() -> Optional[Any]:
    """
    Return a LlamaIndex embedding object or None if unavailable.
    Tries OpenAI if key present, else HuggingFace. Returns None (and warns) if neither works.
    """
    # Try OpenAI first
    try:
        oai_key = (
            st.secrets.get("openaikey", {}).get("key")
            if "openaikey" in st.secrets else os.getenv("OPENAI_API_KEY")
        )
        if oai_key:
            os.environ["OPENAI_API_KEY"] = oai_key
            from llama_index.embeddings.openai import OpenAIEmbedding
            return OpenAIEmbedding(model="text-embedding-3-small")
    except Exception as e:
        st.warning(f"OpenAI embeddings unavailable: {e}")

    # Fallback: HuggingFace (requires model download once)
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.warning(f"HuggingFace embeddings unavailable: {e}")

    st.warning("No embedding backend available. Skipping vector indexing.")
    return None

def _chroma_vector_store(collection_name: str, embed_dim: int) -> ChromaVectorStore:
    """
    Let LlamaIndex create/get the collection. Don’t pre-create it yourself.
    Suffix with embedding dimension to avoid schema conflicts.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    clean_name = collection_name.replace("/", "_").replace(" ", "_")
    final_name = f"{clean_name}_d{embed_dim}"
    # Do NOT call client.get_or_create_collection() yourself; let LlamaIndex manage it.
    return ChromaVectorStore(chroma_client=client, collection_name=final_name)


def normalize_to_json_string(blob) -> str:
    """Ensure blob is always returned as a JSON-formatted string."""
    if isinstance(blob, (dict, list)):
        return json.dumps(blob, indent=2)
    elif isinstance(blob, str):
        try:
            parsed = json.loads(blob)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return json.dumps({"raw_string": blob}, indent=2)
    else:
        return json.dumps({}, indent=2)


def embed_string_for_indexing(content: str, collection_name: str):
    if not isinstance(content, str) or not content.strip():
        st.error("Lab content is empty or not a string.")
        return None

    try:
        embed_model, dim = _resolve_embedding_or_raise()
    except Exception as e:
        st.error(f"Embeddings unavailable: {e}")
        return None

    document = Document(text=content, metadata={"collection": collection_name})
    vector_store = _chroma_vector_store(f"lab_{collection_name}", embed_dim=dim)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)

    # VectorStoreIndex will use Settings.embed_model (we still pass explicitly for clarity)
    index = VectorStoreIndex.from_documents(
        [document],
        storage_context=storage_ctx,
        embed_model=embed_model,
    )
    st.success(f"Embedded and indexed '{collection_name}' (dim={dim})")
    return index


def embed_and_index_logs(log_chunks: Dict[str, Dict[str, List[dict]]], log_name: str):
    if not isinstance(log_chunks, dict):
        st.error("log_chunks must be a dict.")
        return None

    docs: List[Document] = []
    for category, by_source in log_chunks.items():
        if not isinstance(by_source, dict):
            continue
        for source, events in by_source.items():
            for e in events or []:
                docs.append(
                    Document(
                        text=json.dumps(
                            {
                                "category": category,
                                "event_source": source,
                                "event_name": e.get("event_name"),
                                "errorCode": e.get("cloud_trail_event", {}).get("errorCode"),
                                "event": e,
                            },
                            indent=2,
                        ),
                        metadata={"category": category, "source": source},
                    )
                )

    if not docs:
        st.warning("No log documents to index.")
        return None

    try:
        embed_model, dim = _resolve_embedding_or_raise()
    except Exception as e:
        st.error(f"Embeddings unavailable: {e}")
        return None

    vector_store = _chroma_vector_store(f"log_{log_name}", embed_dim=dim)
    storage_ctx = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_ctx,
        embed_model=embed_model,
    )
    return index

# --- Main module
def feedback_analysis_module():
    st.markdown("---")

    # Initialize once
    for k, v in [
        ("feedback_text", ""),
        ("gemini_feedback_analysis", None),
        ("final_feedback_response", None),
    ]:
        st.session_state.setdefault(k, v)

    # Load example phrases (optional)
    bucket, s3 = get_s3()
    sample_responses: List[str] = []
    if bucket and s3:
        try:
            sample_file_key = "sampleresponses/mailresponses.txt"
            sample_content = s3_get_bytes(s3, bucket, sample_file_key)
            if sample_content:
                sample_responses = sample_content.decode("utf-8", errors="replace").split("\n\n")
        except Exception as e:
            st.warning(f"Could not load sample responses: {e}")

    st.session_state["feedback_text"] = st.text_area(
        "Enter Learner's Feedback", key="user_feedback"
    )

    # Surface what's loaded
    with st.expander("Inputs detected", expanded=False):
        st.write("**Has lab content:**", bool(st.session_state.get("lab_content")))
        lab_content = st.session_state.get("lab_content")
        st.write(f"The type of content is {type(lab_content)}")

        st.write("**Has CloudTrail logs (raw JSON text):**", bool(st.session_state.get("log_content")))
        log_content_raw = st.session_state.get("log_content")
        try:
            log_content2 = json.loads(log_content_raw) if isinstance(log_content_raw, str) else log_content_raw
        except json.JSONDecodeError:
            st.error("Could not parse log_content as JSON.")
            log_content2 = None
        st.write(f"The type of log_content is: {type(log_content2)}")

        st.write("**Has policy JSON:**", isinstance(st.session_state.get("policy_json"), (dict, list, str)))
        policy = st.session_state.get("policy_json")
        st.write(f"The type of policy is: {type(policy)}")

        st.write("**Has CloudFormation JSON (prescript):**", isinstance(st.session_state.get("cloudformation_json"), (dict, list, str)))
        cfn = st.session_state.get("cloudformation_json")
        st.write(f"The type of cfn is: {type(cfn)}")

    # -------------------- EVERYTHING BELOW STAYS INSIDE THE BUTTON BLOCK --------------------
    if st.button("Generate email & DRI report", type="primary"):
        # Validate inputs
        if not st.session_state.get("feedback_text"):
            st.error("Please enter learner feedback first.")
            return
        if not st.session_state.get("log_content") or not st.session_state.get("lab_content"):
            st.error("Please ensure both CloudTrail logs and lab content are loaded (use the buttons in S3 sections).")
            return

        # 1) Parse logs ONLY
        try:
            raw = st.session_state["log_content"]
            parsed_logs = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            st.error(f"CloudTrail logs are not valid JSON. {e}")
            return

        if not isinstance(parsed_logs, dict):
            st.error("CloudTrail logs must be a JSON object with an 'events' array.")
            return

        all_events = parsed_logs.get("events")
        if not isinstance(all_events, list):
            st.error("CloudTrail JSON is missing 'events' (array).")
            return

        # 2) Build summaries
        organized_logs: Dict[str, List[dict]] = {}
        for ev in all_events:
            if not isinstance(ev, dict):
                continue
            src = (ev.get("event_source") or "").strip()
            organized_logs.setdefault(src, []).append(ev)

        summarized_logs: Dict[str, List[dict]] = {}
        skip_sources = ["guardduty","devops-guru","health","ce","cost-optimization","notifications","freetier","ram","organizations"]
        for source, events in organized_logs.items():
            if any(x in (source or "").lower() for x in skip_sources):
                continue
            errs = [e for e in events if isinstance(e, dict) and e.get("cloud_trail_event", {}).get("errorCode")]
            for e in errs:
                if "cloud_trail_event" in e:
                    e["cloud_trail_event"].pop("userAgent", None)
                    e["cloud_trail_event"].pop("readOnly", None)
            if errs:
                summarized_logs[source] = errs

        creation_logs: Dict[str, List[dict]] = {}
        skip_sources_creation = ["guardduty","devops-guru","health","ce","cost-optimization","notifications","freetier"]
        for source, events in organized_logs.items():
            if any(x in (source or "").lower() for x in skip_sources_creation):
                continue
            creates = [e for e in events if any(x in (e.get("event_name") or "") for x in ("Create", "Authorize","Run","Publish","Put","Register","Attach", "Update"))]
            if creates:
                creation_logs[source] = creates

        log_chunks = {"error_events": summarized_logs, "creation_events": creation_logs}

        # 3) Indexing (separate try, separate messaging)
        try:
            lab_index = embed_string_for_indexing(st.session_state["lab_content"], "standard")
            st.session_state["lab_index"] = lab_index  # may be None if embeddings unavailable
        except Exception as e:
            #st.error(f"indexing error at content processing: {e}")
            print("without indexes")
            lab_index = None
        try:
            log_index = embed_and_index_logs(log_chunks, "sample_logs2")
            st.session_state["log_index"] = log_index
        except Exception as e:
            #st.error(f"Vector indexing failed in logs: {e}")
            print("without indexes")
            log_index = None

        # 4) Retrieval (with robust fallbacks)
        log_snippets = ""
        lab_snippets = ""

        if log_index:
            try:
                retrieved_log_chunks = log_index.as_retriever(similarity_top_k=15).retrieve(
                    "execution issues, failed events, permission errors, AccessDenied, Create*, Run*, errorCode, Authorize"
                )
                log_snippets = "\n---\n".join([n.get_content() for n in retrieved_log_chunks]).strip()
            except Exception as e:
                st.warning(f"Log retrieval failed; using fallback. {e}")

        if not log_snippets:
            log_snippets = json.dumps(log_chunks, indent=2)

        if lab_index:
            try:
                retrieved_lab_chunks = lab_index.as_retriever(similarity_top_k=5).retrieve(
                    "lab expectations, required resources, region requirements, naming"
                )
                lab_snippets = "\n---\n".join([n.get_content() for n in retrieved_lab_chunks]).strip()
            except Exception as e:
                st.warning(f"Lab retrieval failed; using fallback. {e}")

        if not lab_snippets:
            lab_snippets = (st.session_state.get("lab_content") or "")

        # 5) Gemini analysis
        try:
            # if "openaikey" not in st.secrets:
            #     st.error("OpenAI API key not found.")
            #     return

            # genai.configure(api_key=st.secrets["openaikey"].get("key"))
            # model="gpt-5"
            if "othersecrets" not in st.secrets or not st.secrets["othersecrets"].get("google_key"):
                st.error("Google API key not found in secrets.toml under [othersecrets].google_key")
                return

            genai.configure(api_key=st.secrets["othersecrets"]["google_key"])
            gmodel = genai.GenerativeModel("gemini-2.5-flash")

            policy_str = normalize_to_json_string(st.session_state.get("policy_json"))
            cfn_str = normalize_to_json_string(st.session_state.get("cloudformation_json"))

            gemini_prompt = f"""
You are a DevOps assistant. Based on the retrieved CloudTrail logs and lab context, provide a detailed analysis including:
- Tasks executed successfully
- Tasks that failed and reasons
- Permissions or dependency issues (consider IAM/CloudFormation)
CloudTrail Log Segments:
{log_snippets}

IAM Policy JSON:
{policy_str}

CloudFormation JSON:
{cfn_str}

Lab Instructions Summary:
{lab_snippets}
""".strip()

            with st.spinner("Analyzing logs and generating Gemini report…"):
                # client = _openai_client()
                # resp = client.responses.create(
                #     model=model,
                #     input=[
                #         {
                #             "role": "developer",
                #             "content": [{"type": "input_text", "text": gpt_dev_prompt}],
                #         },
                #         {
                #             "role": "user",
                #             "content": [{"type": "input_text", "text": gemini_prompt}],
                #         },
                #     ],
                # )
                # gresp = ""
                # if getattr(resp, "output_text", None):
                #     gresp = resp.output_text
                gresp = gmodel.generate_content(gemini_prompt)

            analysis_text = (getattr(gresp, "text", None) or "").strip()
            if not analysis_text:
                st.error("Gemini returned an empty response.")
                return

            st.session_state["gemini_feedback_analysis"] = analysis_text

        except Exception as e:
            st.error(f"Gemini analysis failed: {e}")
            return

        # 6) OpenAI generation of learner + DRI responses
        try:
            client = _openai_client()
            if not client:
                return

            style_samples = "\n\n".join(sample_responses)

            final_prompt = f"""You are an assistant helping AWS learners. Based on the detailed analysis given below and learner feedback, generate a personalized response for the user like the samples attached below and a detailed report for the DRI (directly responsible individual for the AWS course).

# Learner Feedback:
{st.session_state['feedback_text']}

# Analysis on the issue:
{st.session_state.get('gemini_feedback_analysis')}

# Sample Past Feedback Responses (output):
{style_samples}

# Output
1. Response for learner
2. Report for DRI
""".strip()


            oai = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "system", "content": final_prompt}],
            )
            out_text = (oai.choices[0].message.content or "").strip()
            if not out_text:
                st.error("OpenAI returned an empty response.")
                return

            st.session_state["final_feedback_response"] = out_text

        except Exception as e:
            st.error(f"OpenAI generation failed: {e}")
            return

        # 7) Persist the final output
        if st.session_state["final_feedback_response"]:
            st.success("Generated learner reply and DRI report.")
            st.download_button(
                "Download final report",
                data=st.session_state["final_feedback_response"].encode("utf-8"),
                file_name="feedback_and_dri_report.txt",
                mime="text/plain",
            )

# Display results if present


# -----------------------------
# Main Application
# -----------------------------

    # st.set_page_config(layout="wide", page_title="Lab Extractor & CloudTrail Analyzer")
    
# Lab Extractor Module
lab_extractor_module()

# CloudTrail Analyzer Module (appears after lab extraction)
st.markdown("---")
cloudtrail_analysis_module()

feedback_analysis_module()


st.subheader("Generated Response")
st.markdown(st.session_state["final_feedback_response"])
