-- ============================================================
-- NGOMeet — Supabase PostgreSQL Schema (Complete)
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- ── MEETINGS TABLE ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.meetings (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title       TEXT        NOT NULL,
    date        DATE        NOT NULL DEFAULT CURRENT_DATE,
    audio_url   TEXT,
    transcript  TEXT,
    summary     TEXT,
    mom_json    JSONB,
    status      TEXT        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','processing','completed','error')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── TASKS TABLE ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.tasks (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    meeting_id  UUID        NOT NULL REFERENCES public.meetings(id) ON DELETE CASCADE,
    user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    description TEXT        NOT NULL,
    assignee    TEXT        DEFAULT 'Unassigned',
    status      TEXT        NOT NULL DEFAULT 'Pending'
                    CHECK (status IN ('Pending','In Progress','Completed')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── PROFILES TABLE ────────────────────────────────────────
-- Stores role and full_name for each user. Linked 1-to-1 with auth.users.
CREATE TABLE IF NOT EXISTS public.profiles (
    id         UUID        PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email      TEXT,
    role       TEXT        NOT NULL DEFAULT 'member'
                               CHECK (role IN ('admin', 'member')),
    full_name  TEXT        NOT NULL DEFAULT 'Member',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── ROW LEVEL SECURITY ───────────────────────────────────────
ALTER TABLE public.meetings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tasks    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- PROFILES RLS POLICIES
-- ============================================================

-- Any logged-in user can read all profiles (needed to display uploader info)
CREATE POLICY "profiles_select_any_authed" ON public.profiles
    FOR SELECT USING (auth.role() = 'authenticated');

-- Users can only update their own profile (e.g. display name changes later)
CREATE POLICY "profiles_update_own" ON public.profiles
    FOR UPDATE USING (auth.uid() = id);

-- Only the service role (backend) can insert profiles (done via trigger below)
-- No INSERT policy needed for anon/authenticated; trigger runs as SECURITY DEFINER.

-- ============================================================
-- MEETINGS RLS POLICIES (Updated for shared visibility)
-- ============================================================

-- SELECT: any authenticated user can read ALL meetings (shared visibility)
CREATE POLICY "meetings_select_any_authed" ON public.meetings
    FOR SELECT USING (auth.role() = 'authenticated');

-- INSERT: any authenticated user can upload a meeting
CREATE POLICY "meetings_insert_any_authed" ON public.meetings
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- UPDATE: only the meeting owner can update their own meeting record
CREATE POLICY "meetings_update_own" ON public.meetings
    FOR UPDATE USING (auth.uid() = user_id);

-- DELETE: meeting owner OR admin
CREATE POLICY "meetings_delete_owner_or_admin" ON public.meetings
    FOR DELETE USING (
        auth.uid() = user_id
        OR EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

-- ============================================================
-- TASKS RLS POLICIES (Updated for shared visibility)
-- ============================================================

-- SELECT: any authenticated user can read all tasks
CREATE POLICY "tasks_select_any_authed" ON public.tasks
    FOR SELECT USING (auth.role() = 'authenticated');

-- INSERT: authenticated users can create tasks (upload pipeline)
CREATE POLICY "tasks_insert_any_authed" ON public.tasks
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- UPDATE: task owner can change status
CREATE POLICY "tasks_update_own" ON public.tasks
    FOR UPDATE USING (auth.uid() = user_id);

-- DELETE: task owner OR admin
CREATE POLICY "tasks_delete_owner_or_admin" ON public.tasks
    FOR DELETE USING (
        auth.uid() = user_id
        OR EXISTS (
            SELECT 1 FROM public.profiles
            WHERE id = auth.uid() AND role = 'admin'
        )
    );

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_meetings_user_id  ON public.meetings(user_id);
CREATE INDEX IF NOT EXISTS idx_meetings_created  ON public.meetings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_meetings_date     ON public.meetings(date DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_meeting_id  ON public.tasks(meeting_id);
CREATE INDEX IF NOT EXISTS idx_tasks_user_id     ON public.tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status      ON public.tasks(status);
CREATE INDEX IF NOT EXISTS idx_profiles_role     ON public.profiles(role);

-- Full-text search on title + summary
CREATE INDEX IF NOT EXISTS idx_meetings_fts ON public.meetings
    USING GIN (to_tsvector('english',
        coalesce(title, '') || ' ' || coalesce(summary, '')
    ));

-- ============================================================
-- AUTO-CREATE PROFILE ON SIGNUP TRIGGER
-- ============================================================
-- Fires after every INSERT into auth.users (i.e., every new signup).
-- Copies the user's email and metadata full_name, assigns default role 'member'.

CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER                    -- Runs with elevated privileges to write to profiles
SET search_path = public
AS $$
DECLARE
    v_full_name TEXT;
BEGIN
    -- Extract full_name from the metadata passed at signup.
    -- raw_user_meta_data is a JSONB column on auth.users.
    v_full_name := TRIM(
        COALESCE(
            NEW.raw_user_meta_data ->> 'full_name',   -- provided at signup
            SPLIT_PART(NEW.email, '@', 1),             -- email prefix fallback
            'Member'                                   -- last resort
        )
    );

    -- Reject empty string (e.g. if email starts with '@')
    IF v_full_name = '' THEN
        v_full_name := 'Member';
    END IF;

    INSERT INTO public.profiles (id, email, role, full_name)
    VALUES (
        NEW.id,
        NEW.email,
        'member',
        v_full_name
    )
    ON CONFLICT (id) DO UPDATE
        SET email     = EXCLUDED.email,
            full_name = COALESCE(EXCLUDED.full_name, profiles.full_name);

    RETURN NEW;
END;
$$;

-- Drop trigger if it already exists (safe re-run)
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_new_user();

-- ============================================================
-- BACKFILL PROFILES FOR EXISTING USERS
-- ============================================================
-- If you already have users in auth.users, create their profile rows now.
INSERT INTO public.profiles (id, email, role, full_name)
SELECT 
    id, 
    email, 
    'member',
    COALESCE(
        TRIM(SPLIT_PART(email, '@', 1)),
        'Member'
    )
FROM auth.users
ON CONFLICT (id) DO UPDATE
SET 
    email = EXCLUDED.email,
    full_name = COALESCE(profiles.full_name, EXCLUDED.full_name);

-- Ensure any remaining NULL full_name gets a default
UPDATE public.profiles
SET full_name = 'Member'
WHERE full_name IS NULL;

-- ============================================================
-- ADMIN PROMOTION EXAMPLES (Run manually as needed)
-- ============================================================
-- To promote a user to admin:
--   UPDATE public.profiles
--   SET role = 'admin'
--   WHERE email = 'admin@yourorg.org';
--
-- To list all admins:
--   SELECT id, email, full_name, role, created_at 
--   FROM public.profiles 
--   WHERE role = 'admin';
--
-- To demote back to member:
--   UPDATE public.profiles 
--   SET role = 'member' 
--   WHERE email = 'person@yourorg.org';
--
-- To update a user's display name:
--   UPDATE public.profiles
--   SET full_name = 'Jane Smith'
--   WHERE email = 'jane@example.org';