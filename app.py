"""
app.py — NGOMeet Flask Application
Features:
  - RBAC: 'admin' and 'member' roles (public.profiles table)
  - Shared meeting visibility; role-gated deletion
  - User full_name stored in session and injected into all templates
  - Manual MoM editing: owners and admins may edit summary + key fields
  - Audio pipeline: local tempfile only (pydub → Groq → Gemini)
  - Shared Tasks visibility with Role-gated status updates
"""

import os
import re
import uuid
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, abort
)
from dotenv import load_dotenv

from utils.supabase_client import get_anon_client, get_service_client
from utils.audio_processor import transcribe_audio_file
from utils.gemini_processor import generate_mom

# ── Bootstrap ─────────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]

ALLOWED_EXTENSIONS = {"mp3", "wav", "ogg", "flac", "m4a", "webm"}
MAX_UPLOAD_MB = 200
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

TEMP_AUDIO_DIR = os.getenv("TEMP_AUDIO_DIR", tempfile.gettempdir())


# ── Template Context ───────────────────────────────────────────────────────────

@app.context_processor
def inject_globals():
    """
    Variables available in every template automatically:
      today           — YYYY-MM-DD for date inputs
      current_role    — 'admin' | 'member' | None
      current_uid     — logged-in user's UUID string
      current_name    — user's full name (from profiles)
    """
    return {
        "today":        datetime.today().strftime("%Y-%m-%d"),
        "current_role": session.get("role"),
        "current_uid":  session.get("user_id"),
        "current_name": session.get("full_name"),
    }


# ── Auth Helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "access_token" not in session or "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    @login_required
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated


def current_user_id() -> str:
    return session.get("user_id", "")


def current_role() -> str:
    return session.get("role", "member")


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _sanitise_name(name: str) -> str:
    """Strip leading/trailing whitespace; collapse inner whitespace."""
    return re.sub(r"\s+", " ", name.strip())


def _fetch_profile(user_id: str) -> dict:
    """
    Fetch role and full_name from public.profiles in a single query.
    Returns {"role": "member", "full_name": ""} on any failure.
    """
    svc = get_service_client()
    try:
        resp = (
            svc.table("profiles")
            .select("role, full_name")
            .eq("id", user_id)
            .single()
            .execute()
        )
        if resp.data:
            return {
                "role":      resp.data.get("role", "member"),
                "full_name": resp.data.get("full_name") or "",
            }
    except Exception as e:
        logger.warning("Could not fetch profile for user %s: %s", user_id, e)
    return {"role": "member", "full_name": ""}


def _can_delete_meeting(meeting_owner_id: str) -> bool:
    """admin → any meeting; member → only their own."""
    return current_role() == "admin" or current_user_id() == meeting_owner_id


def _can_edit_meeting(meeting_owner_id: str) -> bool:
    """Same permission shape as delete: admin or owner."""
    return current_role() == "admin" or current_user_id() == meeting_owner_id


# ── Auth Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        try:
            client  = get_anon_client()
            resp    = client.auth.sign_in_with_password(
                {"email": email, "password": password}
            )
            user_id = resp.user.id

            # Single DB call: fetch both role AND full_name
            profile = _fetch_profile(user_id)

            session["access_token"]  = resp.session.access_token
            session["refresh_token"] = resp.session.refresh_token
            session["user_id"]       = user_id
            session["user_email"]    = resp.user.email
            session["role"]          = profile["role"]
            session["full_name"]     = profile["full_name"] or email.split("@")[0]

            logger.info(
                "User '%s' (%s) logged in as %s",
                session["full_name"], email, profile["role"],
            )
            return redirect(url_for("dashboard"))

        except Exception as e:
            flash(f"Login failed: {str(e)}", "error")

    return render_template("auth/login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        full_name = _sanitise_name(request.form.get("full_name", ""))
        email     = request.form.get("email", "").strip()
        password  = request.form.get("password", "")
        confirm   = request.form.get("confirm_password", "")

        # ── Validation ────────────────────────────────────────────────────────
        errors = []
        if not full_name:
            errors.append("Full name is required.")
        elif len(full_name) < 2:
            errors.append("Full name must be at least 2 characters.")
        if not email:
            errors.append("Email address is required.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")

        if errors:
            for err in errors:
                flash(err, "error")
            # Re-render with the values the user already typed
            return render_template(
                "auth/signup.html",
                prefill={"full_name": full_name, "email": email},
            )

        try:
            client = get_anon_client()
            resp   = client.auth.sign_up({
                "email":    email,
                "password": password,
                # Pass full_name in metadata so the DB trigger can read it
                # from raw_user_meta_data ->> 'full_name'
                "options": {
                    "data": {"full_name": full_name}
                },
            })

            if resp.user:
                # The trigger handle_new_user() picks up full_name from
                # raw_user_meta_data and writes it to public.profiles.
                logger.info("New signup: %s (%s)", full_name, email)
                flash(
                    f"Welcome, {full_name}! Check your email to confirm "
                    "your account, then log in.",
                    "success",
                )
                return redirect(url_for("login"))
            else:
                flash("Signup failed. Please try again.", "error")

        except Exception as e:
            flash(f"Signup failed: {str(e)}", "error")
            return render_template(
                "auth/signup.html",
                prefill={"full_name": full_name, "email": email},
            )

    return render_template("auth/signup.html", prefill={})


@app.route("/logout")
def logout():
    name = session.get("full_name", "")
    try:
        get_anon_client().auth.sign_out()
    except Exception:
        pass
    session.clear()
    flash(f"{'See you, ' + name + '!' if name else 'You have been logged out.'}", "info")
    return redirect(url_for("login"))


# ── Profile Settings ───────────────────────────────────────────────────────────

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile_settings():
    """Allow users to update their own display name."""
    uid = current_user_id()
    svc = get_service_client()

    if request.method == "POST":
        new_name = _sanitise_name(request.form.get("full_name", ""))

        if len(new_name) < 2:
            flash("Name must be at least 2 characters.", "error")
            return redirect(url_for("profile_settings"))

        try:
            svc.table("profiles").update({"full_name": new_name}).eq("id", uid).execute()
            session["full_name"] = new_name
            flash("Display name updated.", "success")
            logger.info("User %s updated their name to '%s'", uid, new_name)
        except Exception as e:
            flash(f"Could not update name: {e}", "error")

        return redirect(url_for("profile_settings"))

    profile_resp = svc.table("profiles").select("full_name, role, email").eq("id", uid).single().execute()
    profile = profile_resp.data or {}
    return render_template("profile.html", profile=profile, user_email=session.get("user_email"))


# ── Dashboard ──────────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    svc = get_service_client()
    uid = current_user_id()

    meetings_resp = (
        svc.table("meetings")
        .select("id, title, date, status, summary, created_at, user_id")
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    meetings = meetings_resp.data or []

    tasks_resp = (
        svc.table("tasks")
        .select("status")
        .eq("user_id", uid)
        .execute()
    )
    tasks = tasks_resp.data or []

    stats = {
        "total_meetings":    len(meetings),
        "pending_tasks":     sum(1 for t in tasks if t["status"] == "Pending"),
        "in_progress_tasks": sum(1 for t in tasks if t["status"] == "In Progress"),
        "completed_tasks":   sum(1 for t in tasks if t["status"] == "Completed"),
    }

    return render_template(
        "dashboard.html",
        meetings=meetings,
        stats=stats,
        user_email=session.get("user_email"),
    )


# ── Upload & Process Meeting ───────────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
@login_required
def upload_meeting():
    """
    Full pipeline — NO Supabase Storage:
      1. Validate form inputs.
      2. Write audio to a local tempfile.
      3. Insert 'processing' DB record immediately (gives instant feedback).
      4. Chunk + transcribe via Groq Whisper (pydub handles splitting).
      5. Generate MoM via Gemini.
      6. Update DB record with transcript / summary / mom_json / tasks.
      7. Delete local tempfile unconditionally in finally block.
    """
    title      = request.form.get("title", "").strip()
    date_str   = request.form.get("date", datetime.today().strftime("%Y-%m-%d"))
    audio_file = request.files.get("audio_file")

    if not title:
        flash("Meeting title is required.", "error")
        return redirect(url_for("dashboard"))
    if not audio_file or audio_file.filename == "":
        flash("Please select an audio file to upload.", "error")
        return redirect(url_for("dashboard"))
    if not _allowed_file(audio_file.filename):
        flash(f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}", "error")
        return redirect(url_for("dashboard"))

    uid        = current_user_id()
    meeting_id = str(uuid.uuid4())
    ext        = audio_file.filename.rsplit(".", 1)[1].lower()
    svc        = get_service_client()

    svc.table("meetings").insert({
        "id": meeting_id, "user_id": uid,
        "title": title, "date": date_str, "status": "processing",
    }).execute()

    tmp_dir  = Path(TEMP_AUDIO_DIR)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{meeting_id}.{ext}"

    try:
        audio_file.seek(0)
        tmp_path.write_bytes(audio_file.read())
        logger.info("Temp audio saved: %s (%d bytes)", tmp_path, tmp_path.stat().st_size)

        logger.info("Transcribing meeting %s …", meeting_id)
        transcript = transcribe_audio_file(str(tmp_path))
        logger.info("Transcription complete: %d chars", len(transcript))

        logger.info("Generating MoM for meeting %s …", meeting_id)
        mom     = generate_mom(transcript)
        summary = mom.get("summary", "")

        svc.table("meetings").update({
            "transcript": transcript,
            "summary":    summary,
            "mom_json":   mom,
            "status":     "completed",
        }).eq("id", meeting_id).execute()

        tasks_payload = [
            {
                "meeting_id":  meeting_id,
                "user_id":     uid,
                "description": t.get("description", ""),
                "assignee":    t.get("assignee", "Unassigned"),
                "status":      "Pending",
            }
            for t in mom.get("tasks", [])
        ]
        if tasks_payload:
            svc.table("tasks").insert(tasks_payload).execute()

        n = len(tasks_payload)
        flash(f"Meeting processed! {n} action item{'s' if n != 1 else ''} extracted.", "success")
        return redirect(url_for("meeting_detail", meeting_id=meeting_id))

    except Exception as e:
        logger.exception("Error processing meeting %s: %s", meeting_id, e)
        try:
            svc.table("meetings").update({
                "status": "error",
                "summary": f"Processing error: {str(e)[:500]}",
            }).eq("id", meeting_id).execute()
        except Exception:
            pass
        flash(f"Processing failed: {str(e)}", "error")
        return redirect(url_for("dashboard"))

    finally:
        try:
            if tmp_path.exists():
                os.remove(tmp_path)
                logger.info("Deleted temp file: %s", tmp_path)
        except OSError as e:
            logger.warning("Could not delete temp file %s: %s", tmp_path, e)


# ── Meeting Detail ─────────────────────────────────────────────────────────────

@app.route("/meeting/<meeting_id>")
@login_required
def meeting_detail(meeting_id):
    svc = get_service_client()

    meeting_resp = (
        svc.table("meetings")
        .select("*")
        .eq("id", meeting_id)
        .single()
        .execute()
    )
    if not meeting_resp.data:
        abort(404)

    meeting = meeting_resp.data

    tasks_resp = (
        svc.table("tasks")
        .select("*")
        .eq("meeting_id", meeting_id)
        .order("created_at")
        .execute()
    )
    tasks = tasks_resp.data or []
    mom   = meeting.get("mom_json") or {}

    # Pre-compute permissions once; templates stay logic-free
    owner_id   = meeting["user_id"]
    can_delete = _can_delete_meeting(owner_id)
    can_edit   = _can_edit_meeting(owner_id)

    # Fetch uploader's display name for the "uploaded by" chip
    uploader_profile = svc.table("profiles").select("full_name").eq("id", owner_id).single().execute()
    uploader_name = (uploader_profile.data or {}).get("full_name") or "Unknown"

    return render_template(
        "meeting_detail.html",
        meeting=meeting,
        tasks=tasks,
        mom=mom,
        can_delete=can_delete,
        can_edit=can_edit,
        uploader_name=uploader_name,
        user_email=session.get("user_email"),
    )


# ── Update Meeting (Manual MoM Edit) ──────────────────────────────────────────

@app.route("/meeting/<meeting_id>/update", methods=["POST"])
@login_required
def update_meeting(meeting_id):
    """
    Save manual corrections to a meeting's AI-generated MoM fields.

    Editable fields (all optional, only present keys are saved):
      - title          : meeting title
      - summary        : freeform text summary
      - key_decisions  : textarea, one decision per line
      - next_meeting_date: ISO date string or blank

    Permission: meeting owner OR admin.
    Responds with JSON (for the in-page AJAX save) or form redirect.
    """
    svc   = get_service_client()
    is_json = request.is_json

    # ── Fetch meeting to verify existence + ownership ─────────────────────────
    check = (
        svc.table("meetings")
        .select("id, user_id, title, summary, mom_json")
        .eq("id", meeting_id)
        .single()
        .execute()
    )
    if not check.data:
        if is_json:
            return jsonify({"error": "Meeting not found"}), 404
        abort(404)

    owner_id = check.data["user_id"]
    if not _can_edit_meeting(owner_id):
        logger.warning(
            "EDIT DENIED — user %s (role=%s) tried to edit meeting %s owned by %s",
            current_user_id(), current_role(), meeting_id, owner_id,
        )
        if is_json:
            return jsonify({"error": "Permission denied"}), 403
        abort(403)

    # ── Parse the submitted fields ────────────────────────────────────────────
    if is_json:
        data = request.get_json(force=True) or {}
    else:
        data = request.form

    updates_meeting: dict = {}     # Goes into the meetings table
    updates_mom:     dict = {}     # Merged back into mom_json JSONB

    # Title
    new_title = (data.get("title") or "").strip()
    if new_title:
        updates_meeting["title"] = new_title

    # Summary — always save even if blank (user may want to clear it)
    if "summary" in data:
        new_summary = (data.get("summary") or "").strip()
        updates_meeting["summary"] = new_summary
        updates_mom["summary"]     = new_summary

    # Key decisions — one per line, blank lines filtered out
    if "key_decisions" in data:
        raw_decisions = data.get("key_decisions", "")
        if isinstance(raw_decisions, str):
            decisions = [d.strip() for d in raw_decisions.splitlines() if d.strip()]
        else:
            decisions = [str(d).strip() for d in raw_decisions if str(d).strip()]
        updates_mom["key_decisions"] = decisions

    # Next meeting date
    if "next_meeting_date" in data:
        nmd = (data.get("next_meeting_date") or "").strip() or None
        updates_mom["next_meeting_date"] = nmd

    # ── Merge mom_json edits into the existing JSONB ──────────────────────────
    if updates_mom:
        existing_mom = check.data.get("mom_json") or {}
        merged_mom   = {**existing_mom, **updates_mom}
        updates_meeting["mom_json"] = merged_mom

    if not updates_meeting:
        if is_json:
            return jsonify({"success": True, "message": "Nothing to update"}), 200
        flash("No changes were submitted.", "info")
        return redirect(url_for("meeting_detail", meeting_id=meeting_id))

    # ── Persist ───────────────────────────────────────────────────────────────
    try:
        svc.table("meetings").update(updates_meeting).eq("id", meeting_id).execute()
        logger.info(
            "Meeting %s updated by user %s (role=%s). Fields: %s",
            meeting_id, current_user_id(), current_role(), list(updates_meeting.keys()),
        )
    except Exception as e:
        logger.exception("Failed to update meeting %s: %s", meeting_id, e)
        if is_json:
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        flash(f"Save failed: {str(e)}", "error")
        return redirect(url_for("meeting_detail", meeting_id=meeting_id))

    if is_json:
        return jsonify({
            "success":  True,
            "message":  "Changes saved.",
            "fields":   list(updates_meeting.keys()),
        })

    flash("Meeting updated successfully.", "success")
    return redirect(url_for("meeting_detail", meeting_id=meeting_id))


# ── Delete Meeting ─────────────────────────────────────────────────────────────

@app.route("/meeting/<meeting_id>/delete", methods=["POST"])
@login_required
def delete_meeting(meeting_id):
    svc = get_service_client()

    check = (
        svc.table("meetings")
        .select("id, user_id, title")
        .eq("id", meeting_id)
        .single()
        .execute()
    )
    if not check.data:
        abort(404)

    meeting_owner_id = check.data["user_id"]
    meeting_title    = check.data["title"]

    if not _can_delete_meeting(meeting_owner_id):
        logger.warning(
            "DELETE DENIED — user %s (role=%s) tried to delete meeting %s owned by %s",
            current_user_id(), current_role(), meeting_id, meeting_owner_id,
        )
        abort(403)

    svc.table("tasks").delete().eq("meeting_id", meeting_id).execute()
    svc.table("meetings").delete().eq("id", meeting_id).execute()

    logger.info(
        "Meeting '%s' (%s) deleted by user %s (role=%s)",
        meeting_title, meeting_id, current_user_id(), current_role(),
    )
    flash("Meeting deleted.", "info")
    return redirect(url_for("dashboard"))


# ── Tasks Board ────────────────────────────────────────────────────────────────

@app.route("/tasks")
@login_required
def tasks_board():
    svc = get_service_client()
    
    # Shared Visibility: We removed .eq("user_id", uid) so everyone sees all tasks across the org
    tasks_resp = (
        svc.table("tasks")
        .select("*, meetings(title, date)")
        .order("created_at", desc=True)
        .execute()
    )
    tasks = tasks_resp.data or []

    pending     = [t for t in tasks if t["status"] == "Pending"]
    in_progress = [t for t in tasks if t["status"] == "In Progress"]
    completed   = [t for t in tasks if t["status"] == "Completed"]

    return render_template(
        "tasks.html",
        pending=pending,
        in_progress=in_progress,
        completed=completed,
        user_email=session.get("user_email"),
    )


@app.route("/api/tasks/<task_id>/status", methods=["PATCH"])
@login_required
def update_task_status(task_id):
    data       = request.get_json(force=True)
    new_status = data.get("status", "")

    if new_status not in ("Pending", "In Progress", "Completed"):
        return jsonify({"error": "Invalid status value"}), 400

    svc = get_service_client()
    uid = current_user_id()
    role = current_role()

    # Base query
    query = svc.table("tasks").update({"status": new_status}).eq("id", task_id)
    
    # Enforce RBAC: If you are NOT an admin, you must own the task to edit it
    if role != "admin":
        query = query.eq("user_id", uid)

    resp = query.execute()

    if not resp.data:
        return jsonify({"error": "Task not found or permission denied"}), 403

    return jsonify({"success": True, "task_id": task_id, "status": new_status})


# ── Archive ────────────────────────────────────────────────────────────────────

@app.route("/archive")
@login_required
def archive():
    svc       = get_service_client()
    query     = request.args.get("q", "").strip()
    date_from = request.args.get("date_from", "")
    date_to   = request.args.get("date_to", "")

    base = svc.table("meetings").select("id, title, date, status, summary, created_at, user_id")

    if query:
        base = base.or_(f"title.ilike.%{query}%,summary.ilike.%{query}%")
    if date_from:
        base = base.gte("date", date_from)
    if date_to:
        base = base.lte("date", date_to)

    resp     = base.order("date", desc=True).limit(100).execute()
    meetings = resp.data or []

    return render_template(
        "archive.html",
        meetings=meetings,
        query=query,
        date_from=date_from,
        date_to=date_to,
        user_email=session.get("user_email"),
    )


# ── Error Handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", code=403,
        message="You do not have permission to perform this action."), 403

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404

@app.errorhandler(413)
def too_large(e):
    flash(f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB.", "error")
    return redirect(url_for("dashboard"))

@app.errorhandler(500)
def server_error(e):
    logger.exception("500 error: %s", e)
    return render_template("error.html", code=500, message="Internal server error."), 500


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1")
    app.run(host="0.0.0.0", port=5000, debug=debug)