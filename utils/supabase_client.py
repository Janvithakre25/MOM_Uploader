"""
utils/supabase_client.py
Singleton Supabase clients — one for anon operations (auth),
one service-role client for backend storage/DB writes that bypass RLS.
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

_anon_client: Client | None = None
_service_client: Client | None = None


def get_anon_client() -> Client:
    """
    Returns a Supabase client using the anon/public key.
    Used for auth operations and user-scoped queries.
    """
    global _anon_client
    if _anon_client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_ANON_KEY"]
        _anon_client = create_client(url, key)
    return _anon_client


def get_service_client() -> Client:
    """
    Returns a Supabase client using the service_role key.
    Bypasses RLS — use ONLY for trusted backend operations
    (uploading to storage, inserting records on behalf of user).
    """
    global _service_client
    if _service_client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]
        _service_client = create_client(url, key)
    return _service_client
