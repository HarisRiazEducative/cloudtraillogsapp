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
            source: [event for event in events if any(x in event.get("event_name", "") for x in ("Create", "Authorize","Run","Publish","Put","Register","Attach"))]
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
    """Main CloudTrail analysis module with S3 file selection and AI analysis."""
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
    st.caption("Select CloudTrail logs from S3 and analyze with AI-powered insights")
    
    # Get S3 connection
    bucket, s3 = get_s3()
    if not bucket or not s3:
        st.error("Missing S3 setup. Add [aws] bucket/region/credentials to secrets.toml.")
        return
    
    # File source selection
    file_source = st.radio(
        "Choose file source",
        options=["Select from S3", "Upload from device"],
        horizontal=True,
        help="Select files from S3 bucket or upload from your device",
        key="cloudtrail_file_source"
    )
    
    if file_source == "Select from S3":
        # S3 file selection for CloudTrail logs
        st.subheader("Select CloudTrail Logs from S3")
        execution_logs = s3_list_keys(bucket, s3, "execution_logs/")
        if not execution_logs:
            st.info("No execution logs found in S3. Please extract some labs first or upload files from device.")
            return
            
        log_files = [item["Key"] for item in execution_logs]
        selected_log_file = st.selectbox("Select CloudTrail log file", options=log_files, key="s3_log_select")
        
        # Load logs button
        if selected_log_file and st.button("Load and Process Logs", key="load_logs_btn"):
            with st.spinner("Loading logs from S3..."):
                log_data = s3_get_bytes(s3, bucket, selected_log_file)
                if log_data:
                    # Store the raw log content in session state
                    st.session_state['log_content'] = log_data.decode('utf-8')
                    
                    # Create a mock uploaded file object
                    class MockUploadedFile:
                        def __init__(self, data):
                            self.data = data
                            self.size = len(data)
                            self.position = 0
                        
                        def read(self, size=None):
                            if size is None:
                                # Read all remaining data
                                remaining = self.data[self.position:]
                                self.position = len(self.data)
                                return remaining
                            else:
                                # Read specific size
                                end_pos = min(self.position + size, len(self.data))
                                chunk = self.data[self.position:end_pos]
                                self.position = end_pos
                                return chunk
                        
                        def seek(self, pos):
                            self.position = pos
                    
                    mock_file = MockUploadedFile(log_data)
                    st.session_state.cloudtrail_organized_logs = load_and_process_logs(mock_file)
                    st.session_state.cloudtrail_selected_log_file = selected_log_file
                    st.success(f"Loaded logs from s3://{bucket}/{selected_log_file}")
                else:
                    st.error("Failed to load log file from S3")
        
        # S3 file selection for resource config
        st.subheader("Select Resource Config from S3")
        resource_configs = s3_list_keys(bucket, s3, "resource_configs/")
        if resource_configs:
            config_files = [item["Key"] for item in resource_configs]
            selected_config_file = st.selectbox("Select resource config file", options=config_files, key="s3_config_select")
            
            # Load config button
            if selected_config_file and st.button("Load Resource Config", key="load_config_btn"):
                config_data = s3_get_bytes(s3, bucket, selected_config_file)
                if config_data:
                    # Load the lab content based on lab name from selected config file
                    lab_name = selected_config_file.split('/')[-1].replace('_resource_config.json', '')
                    lab_content_key = f"content_docs/{lab_name}.txt"
                    lab_content = s3_get_bytes(s3, bucket, lab_content_key)
                    if lab_content:
                        st.session_state['lab_content'] = lab_content.decode('utf-8', errors='replace')
                
                    # Store the config content
                    st.session_state.cloudtrail_lab_spec_content = config_data.decode("utf-8", errors="replace")
                    st.session_state.cloudtrail_selected_config_file = selected_config_file
                    st.success(f"Loaded resource config from s3://{bucket}/{selected_config_file}")
                    st.json(json.loads(st.session_state.cloudtrail_lab_spec_content))
                else:
                    st.error("Failed to load resource config from S3")
        else:
            st.info("No resource configs found in S3. Please generate some resource configs first.")
    
    else:  # Upload from device
        st.subheader("Upload Files from Device")
        uploaded_file = st.file_uploader("Upload CloudTrail JSON Log File", type="json", key="cloudtrail_uploader")
        
        if uploaded_file:
            with st.spinner("Processing logs..."):
                st.session_state.cloudtrail_organized_logs = load_and_process_logs(uploaded_file)
        
        lab_spec_file = st.file_uploader("Upload Lab Specification (JSON)", type="json", key="lab_spec")
        if lab_spec_file:
            try:
                st.session_state.cloudtrail_lab_spec_content = lab_spec_file.read().decode("utf-8")
                st.success("Lab specifications loaded.")
            except:
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
            st.markdown("### 🔍 Analyze Errors in the labs")
            if st.session_state.cloudtrail_lab_spec_content:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Analyze Lab Issues", key="analyze_lab_btn"):
                        if "gemini_lab_analysis" not in st.session_state:
                            try:
                                asyncio.run(analyze_lab_with_gemini(st.session_state.cloudtrail_organized_logs, st.session_state.cloudtrail_lab_spec_content))
                            except Exception as e:
                                if "429" in str(e):
                                    st.error("⚠️ Gemini API quota exceeded. Please wait a moment and try again, or check your API usage limits.")
                                else:
                                    st.error(f"API error: {e}")
            else:
                st.info("Please select or upload a resource config file to enable lab analysis.")

            if "gemini_lab_analysis" in st.session_state:
                st.markdown("### 🔍 Analysis Results")
                st.markdown(st.session_state["gemini_lab_analysis"])
                
        elif selected_tab == "Resource Compliance":
            st.markdown("### 🛠 Resource Creation Analysis")
            if st.session_state.cloudtrail_lab_spec_content:
                if st.button("Check Resource Creation Compliance", key="analyze_compliance_btn"):
                    if "gemini_resource_analysis" not in st.session_state:
                        try:
                            asyncio.run(analyze_resource_creation(st.session_state.cloudtrail_organized_logs, st.session_state.cloudtrail_lab_spec_content))
                        except Exception as e:
                            if "429" in str(e):
                                st.error("⚠️ Gemini API quota exceeded. Please wait a moment and try again, or check your API usage limits.")
                            else:
                                st.error(f"Gemini API error: {e}")
                if "gemini_resource_analysis" in st.session_state:
                    st.markdown(st.session_state["gemini_resource_analysis"])
            else:
                st.info("Please select or upload a resource config file to enable compliance analysis.")
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

def parse_editor_ids(editor_url: str) -> Optional[Tuple[str, str, str]]:
    m = EDITOR_IDS_RE.search(editor_url or "")
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)

def build_events_api_url(editor_url: str, user_id: str, attempt_id: str, region: str) -> Optional[str]:
    ids = parse_editor_ids(editor_url)
    if not ids:
        return None
    id1, id2, id3 = ids
    region = (region or "").strip()
    user_id = (user_id or "").strip()
    attempt_id = (attempt_id or "").strip()
    if not (region and user_id and attempt_id):
        return None
    return f"https://www.educative.io/api/cloud-lab/{id1}/{id2}/{id3}/events?user_id={user_id}&attempt_id={attempt_id}&region={region}"

def get_root_url_from_content_file(s3, bucket: str, lab: str) -> Optional[str]:
    key = f"content_docs/{lab}.txt"
    data = s3_get_bytes(s3, bucket, key)
    if not data:
        return None
    text = data.decode("utf-8", errors="replace")
    m = re.search(r"^Root URL:\s*(\S+)", text, flags=re.MULTILINE)
    return m.group(1) if m else None

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

def cloudtrail_section_ui(*, bucket: str, s3, lab_name: str, default_editor_url: Optional[str], cookie_header: str):
    """Renders the Execution Logs UI and writes output to S3."""
    st.divider()
    st.subheader("Execution Logs (CloudTrail-like)")

    if not lab_name:
        st.info("Select or create a lab first to attach execution logs.")
        return

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.caption("Editor URL used to derive the API endpoint")
        editor_url = st.text_input(
            "Lab Editor URL",
            value=default_editor_url or "",
            placeholder="https://www.educative.io/editor/cloudlabeditor/<id1>/<id2>/<id3>",
            key=f"editor_url_{lab_name}"
        )
    with col_b:
        ids = parse_editor_ids(editor_url or "")
        st.write("IDs parsed:" + (f" {ids}" if ids else " —"))

    col1, col2, col3 = st.columns(3)
    with col1:
        user_id = st.text_input("User ID", key=f"user_{lab_name}")
    with col2:
        attempt_id = st.text_input("Attempt ID", key=f"attempt_{lab_name}")
    with col3:
        region = st.text_input("Region", value="us-east-1", key=f"region_{lab_name}")

    # ---- NEW: Attempts history UI
    hist_col1, hist_col2 = st.columns([1, 2])
    with hist_col1:
        run_hist = st.button("Check attempts history", key=f"check_hist_{lab_name}")
    with hist_col2:
        st.caption("Looks up the user's attempts for this lab, so you can pick the correct attempt_id.")

    if run_hist:
        if not (editor_url and user_id):
            st.error("Please provide Editor URL and User ID to check history.")
        else:
            history_url = build_history_api_url(editor_url, user_id)
            if not history_url:
                st.error("Could not build the history API URL. Check the editor URL and inputs.")
            else:
                with st.spinner("Fetching attempts history…"):
                    try:
                        history_json = fetch_attempt_history(history_url, cookie_header)
                    except Exception as e:
                        st.error(f"Failed to fetch history: {e}")
                        history_json = None

                if history_json is not None:
                    attempts = parse_attempts_from_history(history_json)
                    count = len(attempts)
                    st.info(f"Found **{count}** attempt(s) for user `{user_id}` in this lab.")
                    if count:
                        st.dataframe(attempts, use_container_width=True)
                        # Let the user pick one
                        choices = [a["attempt_id"] for a in attempts if a.get("attempt_id")]
                        default_idx = 0 if choices else None
                        # chosen = st.selectbox("Choose an attempt to use", options=choices or [""],
                        #                       index=default_idx if default_idx is not None else 0,
                        #                       key=f"attempt_pick_{lab_name}")
                        # if chosen:
                        #     use_btn = st.button("Use selected attempt", key=f"use_attempt_{lab_name}")
                        #     if use_btn:
                        #         # Autofill Attempt ID and Region (if available)
                        #         picked = next((a for a in attempts if a.get("attempt_id") == chosen), None)
                        #         st.session_state[f"attempt_{lab_name}"] = chosen
                        #         if picked and picked.get("region"):
                        #             st.session_state[f"region_{lab_name}"] = picked["region"]
                        #         st.success(f"Using attempt_id={chosen}" + (f", region={picked.get('region')}" if picked and picked.get("region") else ""))

                        #         # Optional: show the exact events URL we’ll hit next
                        #         api_url = build_events_api_url(editor_url, user_id, chosen, st.session_state.get(f"region_{lab_name}", region))
                        #         if api_url:
                        #             st.caption("Events API URL:")
                        #             st.code(api_url, language="text")

    # ---- Existing logs in S3, unchanged
    run = st.button("Fetch & Save Execution Logs to S3", type="primary", key=f"fetch_logs_{lab_name}")

    existing = list_execution_logs_for_lab(s3, bucket, lab_name)
    with st.expander("Existing logs for this lab in S3"):
        if existing:
            key_choice = st.selectbox("Pick a saved logs file to preview", options=existing, index=0, key=f"existing_logs_{lab_name}")
            if key_choice:
                st.markdown("**Preview**")
                view_s3_file_inline(s3, bucket, key_choice, preview_chars=4000)
                selected = st.button("Use this logs file for analysis", key=f"use_logs_{lab_name}")
                if selected:
                    st.session_state.setdefault("selected_lab_files", {})
                    st.session_state["selected_lab_files"]["execution_logs_key"] = key_choice
                    # NEW: load into session so generators can read it
                    data = s3_get_bytes(s3, bucket, key_choice)
                    if data:
                        st.session_state["log_content"] = data.decode("utf-8", errors="replace")
                        st.success(f"Selected & loaded logs: {key_choice}")
                    else:
                        st.warning("Could not load the selected logs file.")
        else:
            st.caption("No logs yet. Fetch and save to create one.")

    if run:
        if not (editor_url and user_id and (st.session_state.get(f"attempt_{lab_name}") or attempt_id) and (st.session_state.get(f"region_{lab_name}") or region)):
            st.error("Please fill all inputs (Editor URL, User ID, Attempt ID, Region). Tip: use 'Check attempts history' above.")
            return

        # Prefer session_state (if user picked an attempt) over the text inputs
        effective_attempt = st.session_state.get(f"attempt_{lab_name}") or attempt_id
        effective_region = st.session_state.get(f"region_{lab_name}") or region

        api_url = build_events_api_url(editor_url, user_id, effective_attempt, effective_region)
        if not api_url:
            st.error("Could not build the events API URL. Check the editor URL and inputs.")
            return

        with st.spinner("Fetching execution logs..."):
            try:
                logs = fetch_execution_logs(api_url, cookie_header)
            except Exception as e:
                st.error(f"Failed to fetch logs: {e}")
                return

        key = f"execution_logs/{sanitize_lab_name(lab_name)}_{user_id}_{attempt_id}.json"
        try:
            s3_put_json(s3, bucket, key, logs)
            st.success(f"Saved logs to s3://{bucket}/{key}")
            view_s3_file_inline(s3, bucket, key, preview_chars=200000)
            st.session_state.setdefault("selected_lab_files", {})
            st.session_state["selected_lab_files"]["execution_logs_key"] = key
        except Exception as e:
            st.error(f"Failed to write logs to S3: {e}")


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
# Main Lab Extraction Module
# -----------------------------
def lab_extractor_module():
    st.header("Lab Content Extractor")
    st.caption("Educative CloudLabs with Playwright + Requests fallback. Outputs are saved to S3 (content_docs / policies / prescript / execution_logs).")

    # --- S3 init
    bucket, s3 = get_s3()
    if not bucket or not s3:
        st.error("Missing S3 setup. Add [aws] bucket/region/credentials to secrets.toml.")
        st.stop()

    # --- Cookie source controls
    default_cookie = (st.secrets.get("cookies", {}) or {}).get("educative", "")
    cookie_source = st.selectbox(
        "Cookie source",
        options=("Use cookie from secrets", "Enter custom cookie"),
        index=0 if default_cookie else 1,
        help="Use a Cookie header from secrets.toml or supply a fresh one."
    )
    custom_cookie = ""
    show_cookie_preview = ""
    if cookie_source == "Use cookie from secrets":
        show_cookie_preview = (default_cookie[:80] + "…") if len(default_cookie) > 80 else default_cookie
        st.text_input("Cookie (from secrets, read-only)", value=show_cookie_preview, disabled=True)
    else:
        custom_cookie = st.text_area("Custom Cookie (paste full `Cookie:` line or just the cookie value)", height=100)

    st.divider()

    # --- Quick S3 Browser
    with st.expander("🔎 Browse S3 files (content_docs / policies / execution_logs)", expanded=False):
        s3_prefix_browser_ui(bucket, s3)

    # --- Mode: use existing OR extract new
    mode = st.radio(
        "What would you like to do?",
        options=("Use existing from S3", "Extract new or refresh"),
        index=0,
        help="Pick an already-saved lab to download/preview, or extract a new lab and save it to S3."
    )

    cookie_header = default_cookie.strip() if cookie_source == "Use cookie from secrets" else custom_cookie.strip()
    if cookie_header and not cookie_header.lower().startswith("cookie"):
        cookie_header = "Cookie: " + cookie_header

    if mode == "Use existing from S3":
        with st.status("Listing labs in S3…", expanded=False):
            labs = s3_list_labs(bucket, s3)
        if not labs:
            st.info("No labs found in S3 yet. Switch to **Extract new or refresh** to add one.")
            st.stop()

        lab_choice = st.selectbox("Choose a saved lab", options=labs, index=0)
        if not lab_choice:
            st.stop()

        content_key = f"content_docs/{lab_choice}.txt"
        policy_key  = f"policies/{lab_choice}.json"
        prescript_key   = f"prescript/{lab_choice}.json"

        colA, colB, colC = st.columns(3)
        with colA:
            if s3_object_exists(s3, bucket, content_key):
                data = s3_get_bytes(s3, bucket, content_key)
                st.download_button("Download content (.txt)", data=data, file_name=f"{lab_choice}.txt", mime="text/plain", key=f"triplet_content_{lab_choice}")
        with colB:
            if s3_object_exists(s3, bucket, policy_key):
                data = s3_get_bytes(s3, bucket, policy_key)
                st.download_button("Download policy (.json)", data=data, file_name=f"{lab_choice}.json", mime="application/json", key=f"triplet_policy_{lab_choice}")
            else:
                st.caption("No policy JSON")
        with colC:
            if s3_object_exists(s3, bucket, prescript_key):
                data = s3_get_bytes(s3, bucket, prescript_key)
                st.download_button("Download prescript (.json)", data=data, file_name=f"{lab_choice}.json", mime="application/json", key=f"triplet_prescript_{lab_choice}")
            else:
                st.caption("No prescript JSON")

        s3_view_lab_triplet_ui(bucket, s3, lab_choice)

        # Execution logs
        derived_root = get_root_url_from_content_file(s3, bucket, lab_choice)
        cloudtrail_section_ui(
            bucket=bucket,
            s3=s3,
            lab_name=lab_choice,
            default_editor_url=derived_root,
            cookie_header=cookie_header
        )

        # Resource config (single-call)
        generate_resource_config_section(bucket=bucket, s3=s3, lab_name=lab_choice)
        return

    # =============== Extract new or refresh ===============
    with st.form("extract_form"):
        col1, col2 = st.columns([2, 1])
        with col1:
            root_url = st.text_input("Lab Landing URL", placeholder="https://www.educative.io/cloudlabs/...", value="")
        with col2:
            lab_name = st.text_input("Lab Name (used for filenames)", placeholder="my_aws_lab", value="")

        max_pages = st.number_input("Max pages (generic mode)", min_value=1, max_value=2000, value=50, step=1)
        same_path_only = st.checkbox("Restrict to same path (generic)", value=True)
        delay = st.number_input("Request delay (s, generic)", min_value=0.0, max_value=5.0, value=0.3, step=0.1)

        st.divider()
        edu_mode = st.checkbox("Educative CloudLabs mode", value=True)
        use_playwright = st.checkbox("Use headless browser (Playwright, recommended)", value=True)
        timeout_ms = st.number_input("Playwright timeout (ms)", min_value=2000, max_value=60000, value=20000, step=1000)
        debug = st.checkbox("Debug output", value=False)

        extra_headers = None
        submitted = st.form_submit_button("Extract & Save to S3")

    if not submitted:
        st.stop()

    if not root_url:
        st.warning("Please enter a URL.")
        st.stop()

    if not lab_name:
        st.warning("Please enter a Lab Name.")
        st.stop()

    parsed = urlparse(root_url)
    if parsed.scheme not in ("http", "https"):
        st.error("URL must start with http:// or https://")
        st.stop()

    # Prefer Playwright for Educative pages
    if edu_mode and use_playwright:
        try:
            ensure_chromium_installed()
            with st.status("Rendering & extracting via Playwright...", expanded=debug) as status:
                try:
                    import subprocess, sys
                    try:
                        from playwright.sync_api import sync_playwright  # noqa
                    except Exception:
                        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
                        from playwright.sync_api import sync_playwright
                    from playwright.sync_api import sync_playwright
                except Exception as e:
                    raise RuntimeError("Playwright not installed. Run: pip install playwright && playwright install chromium") from e

                links: List[TaskLink] = []
                arts: List[TaskArtifact] = []
                landing_artifact: Optional[TaskArtifact] = None

                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    context = browser.new_context(user_agent=USER_AGENT, ignore_https_errors=True)
                    cookie_value = ""
                    if cookie_header:
                        line = cookie_header
                        if ":" in line:
                            k, v = line.split(":", 1)
                            if k.strip().lower() == "cookie":
                                cookie_value = v.strip()
                        else:
                            cookie_value = line.strip()
                    if cookie_value:
                        context.add_cookies(parse_cookie_header_to_playwright_cookies(cookie_value, root_url))

                    page = context.new_page()
                    page.set_default_timeout(int(timeout_ms))

                    # LANDING
                    page.goto(root_url, wait_until="domcontentloaded")
                    try:
                        page.wait_for_load_state("networkidle")
                    except Exception:
                        pass

                    landing_slate = pw_get_slate_text(page)
                    ace_vals = pw_get_all_ace_values(page)
                    landing_blocks: List[Dict[str, str]] = [{"type": "ace", "content": v} for v in ace_vals if str(v).strip()]
                    landing_title = page.title() or root_url
                    landing_artifact = TaskArtifact(url=root_url, title=landing_title, slate_text=landing_slate, code_blocks=landing_blocks)

                    # Collect task links
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

                    # PER-TASK PAGES
                    for tl in links:
                        try:
                            page.goto(tl.href, wait_until="domcontentloaded")
                            try:
                                page.wait_for_load_state("networkidle")
                            except Exception:
                                pass
                            slate_text = pw_get_slate_text(page)
                            per_task_code_vals = pw_get_all_ace_values(page)
                            code_blocks: List[Dict[str, str]] = [{"type": "ace", "content": v} for v in per_task_code_vals if str(v).strip()]
                            title = tl.title or (page.title() or tl.href)
                            arts.append(TaskArtifact(url=tl.href, title=title, slate_text=slate_text, code_blocks=code_blocks))
                            time.sleep(0.2)
                        except Exception:
                            arts.append(TaskArtifact(url=tl.href, title=tl.title or tl.href, slate_text="", code_blocks=[]))

                    context.close()
                    browser.close()

                status.update(label=f"Playwright extracted {len(links)} task link(s)", state="complete")

            # Write three files to S3
            content_key, policy_key, prescript_key = s3_write_outputs_to_folders(
                s3=s3,
                bucket=bucket,
                lab_name=lab_name,
                root_url=root_url,
                landing=landing_artifact,
                tasks=arts,
            )

            st.success("Saved files to S3.")
            content_bytes = s3_get_bytes(s3, bucket, content_key)
            if content_bytes:
                st.download_button("Download lab content (.txt)", data=content_bytes, file_name=pathlib.Path(content_key).name, mime="text/plain")

            if policy_key and s3_object_exists(s3, bucket, policy_key):
                policy_bytes = s3_get_bytes(s3, bucket, policy_key)
                st.download_button("Download policy (.json)", data=policy_bytes, file_name=pathlib.Path(policy_key).name, mime="application/json")
            else:
                st.info("No policy JSON found on the landing page (first code block).")

            if prescript_key and s3_object_exists(s3, bucket, prescript_key):
                presc_bytes = s3_get_bytes(s3, bucket, prescript_key)
                st.download_button("Download prescript (.json)", data=presc_bytes, file_name=pathlib.Path(prescript_key).name, mime="application/json")
            else:
                st.info("No prescript/template JSON found on the landing page (second code block).")

            st.divider()
            st.markdown("### View saved files")
            lab_sanitized = sanitize_lab_name(lab_name)
            s3_view_lab_triplet_ui(bucket, s3, lab_sanitized)

            # Execution Logs
            cloudtrail_section_ui(
                bucket=bucket,
                s3=s3,
                lab_name=lab_sanitized,
                default_editor_url=root_url,
                cookie_header=cookie_header
            )

            # Resource config (single-call)
            generate_resource_config_section(bucket=bucket, s3=s3, lab_name=lab_sanitized or lab_name)
            return
        except Exception as e:
            st.error(f"Playwright mode failed: {e}")
            st.info("Run: pip install playwright && playwright install chromium. You can also try Requests fallback below.")

    # -----------------------------
    # Educative (Requests fallback)
    # -----------------------------
    session = build_session(extra_headers)

    if edu_mode and not use_playwright:
        st.subheader("Educative (Requests fallback)")
        if cookie_header:
            cookie_val = cookie_header
            if ":" in cookie_val:
                _, cookie_val = cookie_val.split(":", 1)
            session.headers["Cookie"] = cookie_val.strip()

        try:
            resp = session.get(root_url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            st.error(f"Failed to fetch landing page: {e}")
            st.stop()

        landing_artifact = educative_extract_task_requests(resp.text, root_url)
        task_links = educative_collect_task_links_requests(root_url, resp.text, debug=False)
        if not task_links:
            st.error("No task links were found. The page likely requires JS. Try Playwright mode with a Cookie.")
            st.stop()

        st.success(f"Found {len(task_links)} task link(s). Fetching content...")
        arts: List[TaskArtifact] = []
        progress = st.progress(0.0)
        for i, tl in enumerate(task_links, start=1):
            try:
                r = session.get(tl.href, timeout=45)
                if r.status_code >= 400:
                    arts.append(TaskArtifact(url=tl.href, title=tl.title or tl.href, slate_text="", code_blocks=[]))
                else:
                    art = educative_extract_task_requests(r.text, tl.href)
                    if tl.title and (not art.title or art.title == tl.href):
                        art.title = tl.title
                    arts.append(art)
            except Exception:
                arts.append(TaskArtifact(url=tl.href, title=tl.title or tl.href, slate_text="", code_blocks=[]))
            progress.progress(i / max(1, len(task_links)))

        content_key, policy_key, prescript_key = s3_write_outputs_to_folders(
            s3=s3,
            bucket=bucket,
            lab_name=lab_name,
            root_url=root_url,
            landing=landing_artifact,
            tasks=arts,
        )

        st.success("Saved files to S3.")
        content_bytes = s3_get_bytes(s3, bucket, content_key)
        if content_bytes:
            st.download_button("Download lab content (.txt)", data=content_bytes, file_name=pathlib.Path(content_key).name, mime="text/plain")
        if policy_key and s3_object_exists(s3, bucket, policy_key):
            st.download_button("Download policy (.json)", data=s3_get_bytes(s3, bucket, policy_key), file_name=pathlib.Path(policy_key).name, mime="application/json")
        else:
            st.info("No policy JSON found on the landing page (first code block).")
        if prescript_key and s3_object_exists(s3, bucket, prescript_key):
            st.download_button("Download prescript (.json)", data=s3_get_bytes(s3, bucket, prescript_key), file_name=pathlib.Path(prescript_key).name, mime="application/json")
        else:
            st.info("No prescript/template JSON found on the landing page (second code block).")

        st.divider()
        st.markdown("### View saved files")
        lab_sanitized = sanitize_lab_name(lab_name)
        s3_view_lab_triplet_ui(bucket, s3, lab_sanitized)

        # Execution Logs
        cloudtrail_section_ui(
            bucket=bucket,
            s3=s3,
            lab_name=lab_sanitized,
            default_editor_url=root_url,
            cookie_header=cookie_header
        )

        # Resource config (single-call)
        generate_resource_config_section(bucket=bucket, s3=s3, lab_name=lab_sanitized or lab_name)
        return

    # -----------------------------
    # Generic crawl (fallback)
    # -----------------------------
    st.subheader("Generic crawl (fallback)")
    with st.status("Crawling & extracting...", expanded=True) as status:
        st.write(f"Root: {root_url}")
        pages = crawl_generic(root_url, session, max_pages=int(max_pages), same_path_only=same_path_only, delay=float(delay))
        status.update(label=f"Parsed {len(pages)} page(s)", state="complete")

    if not pages:
        st.error("No pages were parsed. Check access or increase the max pages.")
    else:
        payload = io.StringIO()
        payload.write("LAB CONTENT EXTRACT (GENERIC)\n")
        payload.write(f"Root URL: {root_url}\n")
        payload.write(f"Total pages: {len(pages)}\n")
        payload.write("=" * 80 + "\n\n")
        for i, (url, text) in enumerate(pages.items(), start=1):
            payload.write(f"[Page {i}] {url}\n")
            payload.write("-" * 80 + "\n")
            payload.write(text.strip() + "\n\n")

        lab = sanitize_lab_name(lab_name or f"generic_{hashlib.sha256(root_url.encode('utf-8')).hexdigest()[:10]}")
        key = f"content_docs/{lab}.txt"
        s3_put_text(s3, bucket, key, payload.getvalue(), "text/plain")
        data = s3_get_bytes(s3, bucket, key)
        st.download_button(label=f"Download {pathlib.Path(key).name}", data=data, file_name=pathlib.Path(key).name, mime="text/plain")

        st.divider()
        st.markdown("### View saved file")
        view_s3_file_inline(s3, bucket, key)

        # Logs section (optional)
        cloudtrail_section_ui(
            bucket=bucket,
            s3=s3,
            lab_name=lab,
            default_editor_url="",
            cookie_header=cookie_header
        )

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
            creates = [e for e in events if any(x in (e.get("event_name") or "") for x in ("Create", "Authorize","Run","Publish","Put","Register","Attach"))]
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
