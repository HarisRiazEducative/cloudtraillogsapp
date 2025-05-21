import streamlit as st
import requests
import json
import pandas as pd
import re
import openai
from openai import OpenAI

st.set_page_config(page_title="CloudTrail Failure Viewer", layout="wide")
st.title("🚨 CloudTrail Deployment Failure Viewer")

st.markdown("""
This app fetches AWS CloudTrail logs from Educative Cloud Labs and surfaces **failures and error messages** during deployment.

🔹 Paste a full API URL, or  
🔹 Enter the Lab URL + User ID + Attempt ID, and we’ll fetch data for you.
""")

# ----------------- USER INPUT MODE -----------------
input_mode = st.radio("Choose Input Mode", ["Full API URL", "Decomposed Components"])
api_url = ""
author_id = collection_id = cloud_lab_id = user_id = attempt_id = ""

if input_mode == "Full API URL":
    full_url = st.text_input("Paste the full API URL", placeholder="https://www.educative.io/api/cloud-lab/...")
    if full_url:
        api_url = full_url.strip()
        match = re.match(
            r".*/cloud-lab/(\d+)/(\d+)/(\d+)/events\?user_id=(\d+)&attempt_id=(\d+)", api_url
        )
        if match:
            author_id, collection_id, cloud_lab_id, user_id, attempt_id = match.groups()
else:
    lab_url = st.text_input("Lab URL", placeholder="https://www.educative.io/api/cloud-lab/...")
    user_id = st.text_input("User ID", placeholder="e.g. 4930460945350656")
    attempt_id = st.text_input("Attempt ID", placeholder="e.g. 1")
    region = st.text_input("Region", value="us-east-1")

    if lab_url and user_id and attempt_id:
        # Match and convert editor or normal cloudlab URLs to API format
        match_editor = re.match(
            r"https://www\.educative\.io/editor/collection/page/(\d+)/(\d+)/(\d+)/cloudlab", lab_url
        )
        match_collection = re.match(
            r"https://www\.educative\.io/collection/page/(\d+)/(\d+)/(\d+)/cloudlab", lab_url
        )
        
        if match_editor or match_collection:
            author_id, collection_id, cloud_lab_id = (match_editor or match_collection).groups()
            lab_url = f"https://www.educative.io/api/cloud-lab/{author_id}/{collection_id}/{cloud_lab_id}"

        # Construct the API URL
        api_url = f"{lab_url}/events?user_id={user_id}&attempt_id={attempt_id}&region={region}"

        # Extract identifiers for further use if needed
        match_api = re.match(r".*/cloud-lab/(\d+)/(\d+)/(\d+)", lab_url)
        if match_api:
            author_id, collection_id, cloud_lab_id = match_api.groups()

# ----------------- DEFAULT COOKIE (used by default) -----------------
DEFAULT_COOKIE_HEADER = json.loads(st.secrets["auth"]["default_cookie_header"])

# ----------------- MANUAL COOKIE FALLBACK -----------------
with st.expander("🔐 Trouble loading? Paste your session cookies", expanded=False):
    flask_auth = st.text_input("flask-auth", type="password")
    flask_session = st.text_input("flask-session", type="password")
    secure_auth = st.text_input("secure-auth", type="password")

def get_cookie_header():
    if flask_auth and flask_session and secure_auth:
        return {
            "Cookie": (
                f"flask-auth={flask_auth}; "
                f"flask-session={flask_session}; "
                f"secure-auth={secure_auth}"
            )
        }
    return DEFAULT_COOKIE_HEADER

# ----------------- API WRAPPER -----------------
def fetch_api(url):
    try:
        res = requests.get(url, headers=get_cookie_header())
        res.raise_for_status()
        return res.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"❗ API request failed: {e}")
        st.info("If this was a 401 or 403, scroll up and paste your session cookies manually.")
        return None

# ----------------- CLOUDTRAIL PARSER -----------------
def parse_event_details(event_raw):
    try:
        event = json.loads(event_raw) if isinstance(event_raw, str) else event_raw
    except json.JSONDecodeError:
        return {"Stack Name": "", "Status": "", "Error Message": "", "Error Code": ""}
    params = event.get("requestParameters", {})
    response = event.get("responseElements", {})
    return {
        "Stack Name": params.get("stackName", "") if isinstance(params, dict) else "",
        "Status": response.get("stackStatus", event.get("stackStatus", "")) if isinstance(response, dict) else "",
        "Error Message": event.get("errorMessage", ""),
        "Error Code": event.get("errorCode", "")
    }

# ----------------- ATTEMPT HISTORY -----------------
if all([author_id, collection_id, cloud_lab_id, user_id]):
    history_url = f"https://www.educative.io/api/cloud-lab/{author_id}/{collection_id}/{cloud_lab_id}/history?user_id={user_id}&limit=3"
    if st.button("📘 Fetch Attempt History"):
        with st.spinner("Getting attempt history..."):
            history_data = fetch_api(history_url)
            if history_data:
                st.markdown(f"### ✅ User Attempt History, total attempts: `{len(history_data)}`")
                for attempt in history_data['history']:
                    st.markdown(f"- Attempt ID: `{attempt['attempt_id']}`   --- details: `{attempt}`")

# ----------------- CLOUDTRAIL LOG FETCH -----------------
if api_url:
    st.markdown(f"🔗 **Log Events From:** `{api_url}`")
    if st.button("📊 Fetch CloudTrail Logs"):
        with st.spinner("Fetching CloudTrail logs..."):
            result = fetch_api(api_url)
            if result and "events" in result:
                events = result["events"]
                parsed = []
                for e in events:
                    row = {
                        "Time": e["event_time"],
                        "Name": e["event_name"],
                        "Username": e["username"],
                    }
                    row.update(parse_event_details(e["cloud_trail_event"]))
                    parsed.append(row)

                df = pd.DataFrame(parsed)
                st.session_state["cloudtrail_df"] = df

                if "Status" in df.columns:
                    def is_success(status):
                        status = str(status).upper()
                        return status.endswith("_COMPLETE") or "SUCCEEDED" in status

                    df["Success Message"] = df["Status"].apply(
                        lambda s: "✅ Success" if is_success(s) else ""
                    )

                    if df["Status"].replace("", pd.NA).isna().all():
                        df.drop(columns=["Status", "Success Message"], inplace=True)

                def highlight_failures(row):
                    failure_keywords = ["FAIL", "DENIED", "UNAUTHORIZED", "ERROR", "EXCEPTION", "ACCESSDENIED"]
                    status = str(row.get("Status", "")).upper()
                    error_code = str(row.get("Error Code", "")).upper()
                    error_msg = str(row.get("Error Message", "")).upper()

                    if any(kw in status for kw in failure_keywords) or \
                       any(kw in error_code for kw in failure_keywords) or \
                       any(kw in error_msg for kw in failure_keywords):
                        return ['background-color: rgba(255, 77, 77, 0.2); font-weight: bold'] * len(row)

                    if "✅ Success" in str(row.get("Success Message", "")):
                        return ['background-color: rgba(0, 200, 100, 0.15); font-weight: bold'] * len(row)

                    return [''] * len(row)

                styled_df = df.style.apply(highlight_failures, axis=1)
                st.session_state["styled_cloudtrail_df"] = styled_df
                st.success(f"✅ Fetched {len(df)} log events.")

# ----------------- DISPLAY LOG TABLE (PERSISTENT) -----------------
if "styled_cloudtrail_df" in st.session_state:
    st.dataframe(st.session_state["styled_cloudtrail_df"], use_container_width=True)

# ----------------- AI DIAGNOSIS REPORT -----------------
with st.expander("🧠 AI Diagnosis Report (Beta)"):
    if "cloudtrail_df" not in st.session_state:
        st.warning("Please fetch CloudTrail logs first.")
    else:
        policy_text = st.text_area("Paste Lab IAM Policy (JSON or text format)", height=150)
        user_feedback = st.text_area("Describe any issues encountered during the lab")

        if st.button("Generate Report from Log Data"):
            with st.spinner("Analyzing logs with GPT..."):
                df = st.session_state["cloudtrail_df"]
                failure_df = df[df.apply(lambda row: any(
                    kw in str(row.get("Status", "")).upper() or
                    kw in str(row.get("Error Code", "")).upper() or
                    kw in str(row.get("Error Message", "")).upper()
                    for kw in ["FAIL", "DENIED", "UNAUTHORIZED", "ACCESSDENIED", "ERROR", "EXCEPTION"]
                ), axis=1)]

                if failure_df.empty:
                    st.info("No failure logs found to analyze.")
                else:
                    error_summary = failure_df[["Time", "Name", "Error Code", "Error Message"]].to_string(index=False)

                    input_data = [
                        {
                            "role": "system",
                            "content": """ 
                            You are an AWS troubleshooting assistant analyzing CloudTrail logs, IAM policies, and user feedback to diagnose IAM-related lab failures. Your job is to:

                            - Identify failed IAM actions from CloudTrail logs
                            - Compare them against the permissions allowed in the IAM policy
                            - Generate a professional, structured report including:
                                • Summary of Failures
                                • Root Cause Analysis
                                • Missing Permissions (in a table, if appropriate)
                                • Recommended Policy Adjustments
                                • Next Steps
                            - Distinguish between critical and non-critical (background) denied actions

                            The output should be in markdown format, written professionally and suitable for sharing with a user or team reviewing lab results.

                            """
                        },
                        {
                            "role": "user",
                            "content": f"Here are failure logs from the CloudTrail lab session:\n```{error_summary}```\n\nHere is the IAM policy applied in the lab:\n```{policy_text}```\n\nThe user reported these issues during execution:\n```{user_feedback}```\n\nBased on the logs, IAM scope, and user feedback, explain:\n- What likely failed and why\n- Whether adjusting permissions might resolve it\n- Or if the lab should be retested manually. Provide specific missing permissions if identifiable."
                        }
                    ]

                    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                    response = client.responses.create(
                        model="o4-mini",
                        input=input_data,
                        text={"format": {"type": "text"}},
                        reasoning={"effort": "medium"},
                        tools=[],
                        store=True
                    )
                    st.markdown("### 📋 Suggested Diagnosis")
                    for item in response.output:
                        if item.type == "message":
                            for content in item.content:
                                if content.type == "output_text":
                                    st.markdown(content.text)
