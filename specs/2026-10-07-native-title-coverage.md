# Native title coverage: claude, codex and clearing

Filed: 2026-10-07
Branch: `feat/native-title-coverage`, stacked on `fix/native-session-titles`
(PR #41) @ 18a80b9. Builds on `specs/2026-10-06-native-session-titles.md`;
where the two differ, this note supersedes it.

## Goal

Consumers should show each observed conversation's existing native name,
and reflect a deliberate removal of that name, across pi, claude and codex.
Only existing metadata is used: no summaries, prompt-derived labels or
model calls.

## Sources

| Backend | Source | Slot preference |
| --- | --- | --- |
| pi | transcript `session_info.name` | single slot |
| claude | transcript `custom-title.customTitle`, `ai-title.aiTitle` | custom, then generated |
| codex | `$CODEX_HOME/session_index.jsonl` `{id, thread_name, updated_at}` | single slot per thread |

Within a slot the latest complete record wins (file order, never
`updated_at`). Across claude's slots the custom title wins regardless of
append order, matching claude's own display order. claude's legacy
`summary` records and agent names are deliberately not sources.

## Name values

- **Name.** A string, with whitespace runs collapsed to one space.
- **Removal.** An explicit blank string: pi `name: ""`, claude
  `customTitle`/`aiTitle: ""`, codex `thread_name: ""`. A codex index
  rewrite that drops a thread also counts (see below).
- **Nothing.** A missing field, a non-string, malformed JSON, an
  unterminated line, a claude record whose `sessionId` is present but
  doesn't match the transcript, or an unreadable or missing source. None
  of these ever erase a stored title.

`agent_harness.native_titles` holds the extraction, slot and preference
rules. After a claude custom-title removal the generated title shows; after
a generated-title removal a custom title is kept. With no name left in any
slot, the result is a removal.

Upstream divergence: codex's own batch lookup (`find_thread_names_by_ids`)
skips blank entries rather than treating them as removals. As requested,
the harness treats an explicit blank as a removal.

## Applying names

The precedence from #41 holds: native names apply only to `origin:
external` sessions whose title is unset or native. Explicit titles (from
create, PATCH, fork or the bridge) and harness-owned sessions are never
changed or cleared.

A removal is stored and published as `title: null, title_source:
"native"`. Consumers fall back to project/id, and a later native name sets
the title again. Generic observations carry no title and never clear
anything. `models.observed_title` is the single rule used both for
materialization and for every outgoing `session.updated` payload, so the
two cannot disagree.

Name changes publish the stored session with only the title changed. They
never touch `status`, `updated_at`, stats or the freshness clock, so they
don't count as conversation activity.

## Transcripts (pi, claude)

- **Live.** Each title record updates its slot. Changes apply to the
  existing session.
- **Before discovery.** A claude title record that precedes the first
  cwd+model record, or a codex index entry that precedes rollout
  discovery, is kept and applied once the line that creates the session
  is processed. It never creates a session itself.
- **Restart.** On the first new line of a transcript, its slots are
  rebuilt from complete lines up to the persisted offset. From offset 0,
  ordered replay supplies them. So a new generated title after a restart
  still loses to the earlier custom title.
- **Incomplete prefix.** A rebuild counts only if it recovers the whole
  consumed prefix. An unreadable file, or one truncated below or
  misaligned with the offset, leaves the transcript unhydrated. Nothing is
  applied, the stored title is kept, and the rebuild is retried on the
  next line.
  - **Unreadable.** The prefix is presumably intact. Once a later read
    succeeds, the recovered name is reconciled with the row, even on an
    ordinary conversation line, exactly once. This covers a startup
    backfill that couldn't read the file.
  - **Truncated or misaligned.** The old content is never re-applied.
    Once the tail realigns with a regrown file, later title records apply
    as usual.
- **Startup backfill.** One pass over persisted offsets, using the
  watched path spelling. Each external pi or claude session that is
  untitled or natively titled gets the name, or removal, from its
  consumed prefix, so a name removed while the harness was down cannot
  come back. Rows are written directly, with no event and no `updated_at`
  bump. The scan filters lines on a byte substring before decoding: about
  0.4 s warm per ~400 MB of transcripts on the development machine.

## Codex name index

- **Configuration.** `ObserverSettings.codex_name_index_path()` reads
  `session_index.jsonl` beside an observed codex sessions root: a
  `.codex/sessions` root, or `$CODEX_HOME/sessions`. With no codex root
  observed it reads nothing, and `--no-observer` disables it too.
  `--codex-name-index PATH` overrides the derivation for custom homes or
  roots. Only that one file is read; `~/.codex` is not watched.
- **Refresh.** One `stat` per freshness tick (10 s). An unchanged file is
  skipped. If the same file grew and the bytes just before the consumed
  offset are unchanged, only the appended complete lines are read.
  Anything else is re-read whole: a replacement (new inode), a
  truncation, a same-size change, or a changed tail. An in-place rewrite
  further back that coincides with growth is not detected; codex never
  does that.
- **Applying names.** Names apply to `codex_<uuid>` external sessions,
  with `codex_resume_id` as the join. A cached name for a thread whose
  rollout isn't discovered yet waits for discovery and never creates a
  session. Harness codex sessions are never touched.
- **Removal by rewrite.** Codex `remove_thread_name_entries` (pinned at
  rust-v0.159.0) writes a temp file and renames it over the index. A
  previously named thread missing after such a replacement is a removal,
  but only if all of the following hold:
  - the previous file was read successfully;
  - the new file has a new inode;
  - the new file reads fully and ends in a complete line;
  - it has no more malformed lines than before (codex's rewrite keeps
    lines it can't parse).

  The baseline is the set of thread IDs with a valid entry in the
  previous read, not every name ever cached. A name retained from an
  older read, for example across a damaged replacement, is never inferred
  removed by a later rewrite. A truncated or in-place rewritten file is
  re-read, but nothing is inferred from it. A missing or unreadable file
  changes nothing, and a file that disappears and reappears starts a
  fresh baseline.
- **Startup.** Every external codex session gets its cached name or
  removal. A stored native title whose thread is absent from the index is
  cleared only when the startup read was complete and contained no
  malformed lines.
- **Limit.** For paginated history, codex keeps names primarily in its
  SQLite thread store and writes this index best-effort. A name that
  exists only in that database is invisible to the harness. No SQLite or
  app-server integration is attempted.

## Limitations

- claude removal depends on claude writing an explicit blank value. pi's
  own "missing `name`" removal form is not treated as a removal.
- External pi or claude sessions whose transcripts have no persisted
  offset (never observed by this harness) aren't backfilled; their
  ordered replay supplies names when observed.
- Codex removal inference cannot tell a deliberate hand-edit (an editor's
  atomic save) from codex's own removal. Both count as removals.
- The startup backfill reads transcripts synchronously at observer
  construction.
- Codex index change detection is stat-based (inode, size, mtime). A
  same-size, in-place rewrite within one filesystem timestamp tick (coarse
  on e.g. ZFS) is not noticed until the file changes again. Codex never
  rewrites the index in place.

## Rollback

There is no new schema field beyond #41's `title_source`. Stored
`title: null, title_source: "native"` rows are valid for #41's code.
Rolling back past #41 follows the procedure in its spec.
