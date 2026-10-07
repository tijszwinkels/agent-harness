# Native conversation names → `Session.title`

Filed: 2026-10-06
Branch: `fix/native-session-titles` (from `main` @ fbd4429)

## Problem

A pi conversation named with `/name`, `--name` or `pi.setSessionName()`
(Companion names its conversations this way) is surfaced by the harness
with `title: null`. Consumers that label sessions `title → project →
id` (the Mattermost bridge, desktop indicators) fall back to the project
folder, so many sessions all show the same folder name. The name is in
the transcript: pi appends `{"type":"session_info","name":...}` on
every rename, and the latest one is the current name. The observer
parsed the record and discarded it.

## Behavior

**Sources.**

| Backend | Record | Status |
| --- | --- | --- |
| pi | `session_info.name` | discovery, live renames, restart hydration, startup backfill |
| claude | `custom-title.customTitle` (`/rename`) | live renames on existing observed sessions only |
| claude | `ai-title` (generated) | not a source: not the user's name for the conversation |
| codex | thread names | not supported: kept in `~/.codex/session_index.jsonl`, outside the observed rollouts |

**Provenance.** `Session.title_source` is `"native"` when the observer
copied the title from one of these sources, else `null` (explicit
title from create/PATCH/fork, or no title). Rows written before this
field existed load as `null`, so their titles count as explicit.

**Precedence** (`native_title_may_replace`, applied in
`merge_observed_session` so the in-memory and SQLite repositories
agree): a native name may set the title only on an `origin: external`
session whose title is unset or itself native. It never replaces an
explicit title, e.g. the bridge's channel name, and never touches a
harness-owned session. PATCHing `title` makes it explicit
(`title_source: null`); later native renames are then ignored. Forks
inherit `title_source` together with an inherited title. The observer
applies the same decision to every published `session.updated` payload
for an existing session, so subscribers see the stored title, never a
native name the row refused.

**Names.** Whitespace runs, including newlines, collapse to one space
(as pi does). Missing, non-string and blank names are ignored. pi itself
treats a blank name as "cleared"; the harness keeps the last valid name
instead of clearing, so the indicator falls back to the project folder
only for sessions that were never named.

**Not activity.** A rename publishes a `session.updated` carrying the
stored session with only `title`/`title_source` changed: `status`,
`updated_at` and stats are untouched, and the observer's freshness
clock is not refreshed. `PiTranscriptRegistry.take_announcement`
ignores title-only changes, because a pi announcement marks the session
`running`. A metadata-only transcript therefore stays idle however
often it is renamed.

**Order.** Names are applied in transcript order. The bounded head
peek, which looks ahead of the offset, carries no title, and an
unterminated last line is never read for a name.

**Restart.**
- Known session: registry hydration seeds the stored native title, so
  later announcements carry it and later renames are applied.
- Unknown session resumed mid-file (offset persisted, row missing): the
  transcript is scanned up to the persisted offset for the latest name.
  The bounded head peek would miss names further into the file.
- Startup backfill: untitled external pi sessions that have a recorded
  `pi_transcript_path` get the latest name before the persisted offset.
  The offset is looked up by the stored resolved path, falling back to
  resolving the watched spellings, e.g. under a symlinked root. Rows
  with no consumed prefix are skipped, because the tail replays their
  names. Later records are tailed normally, which keeps renames in
  order. The
  row is written directly, with no event and no `updated_at` bump, like
  the codex resume-id backfill. The scan filters lines on a byte
  substring before decoding JSON (~0.3 s warm for ~400 MB of
  transcripts).

## Limitations

- claude: no backfill and no name on first discovery. A `custom-title`
  appearing before the session exists, or behind the persisted offset,
  is not applied until the next rename record.
- External pi rows without `pi_transcript_path` (written before that
  field existed) are not backfilled.
- Native renames are not applied to harness-owned sessions, even
  untitled ones.
- Clearing a pi name does not clear the harness title.
- Rollback: `Session` validates with `extra="forbid"`, and this
  revision writes `title_source` (including `null`) into every session
  row it saves. Before rolling back to an older harness, restore a
  pre-deploy database backup, or strip the key, e.g.
  `update sessions set payload = json_remove(payload, '$.title_source')`
  with the harness stopped. Earlier added fields carry the same hazard.
