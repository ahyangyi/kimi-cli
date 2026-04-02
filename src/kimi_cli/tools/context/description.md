Recall details from previously compacted conversation context for the current conversation trajectory.

Use this tool when the compaction summary is not enough and you need older context details without reading raw archive files directly.

What it does:
- Lists available compacted-context archives for the current trajectory
- Searches archived pre-compaction messages by targeted keywords
- Returns small, relevant excerpts instead of the whole archive

Guidelines:
- Prefer specific queries such as file paths, function names, error strings, IDs, or distinctive keywords
- Leave `query` empty to list available archives and their short summaries first
- Use `archive_id` to narrow the search when you already know which archive is relevant
- This tool only reads archives created by compaction for the current trajectory
- Returned excerpts are sanitized and may omit hidden thinking content
