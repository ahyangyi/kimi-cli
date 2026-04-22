
---

The above is a list of messages in an agent conversation. You are now given a task to compact this conversation context according to specific priorities and rules.

**Compression Priorities (in order):**
1. **Task State**: A hierarchical view of what the task is, from the highest level goal to the immediate task at hand
2. **System Context**: Project structure, locations of important documents, proper commands / environment variables / keys / tokens for working
3. **Design Decisions**: Architectural choices and their rationale
4. **TODO Items**: Unfinished tasks and known issues

**Compression Rules:**
- MUST KEEP: Description for any task, including both current, ongoing and overarching ones. You should try to keep all details in task description, and do not discard any detail.
- MERGE: Similar discussions into single summary points
- REMOVE: Redundant explanations, failed attempts (keep lessons learned), verbose comments
- CONDENSE: Long code blocks → keep signatures; if code is on disk you should note its path; otherwise you might also include its key logic

**Special Handling:**
- For code: Keep full version if < 20 lines, otherwise keep signature + path / key logic
- For errors: Keep full error message if the error still exists; otherwise discard entirely
- For discussions: Merge repeated items, but do not discard any detail
- For skills: if a skill is immediately useful, include its full content VERBATIM, including a header stating which skill it is. Otherwise DISCARD completely. There is no middle ground.

**Required Output Structure:**

<task_state>
[Highest-level task state, including a brief of what was done, what's ongoing and what needs to be done
</task_satte>

<task_state>
[Second-level task state]
</task_satte>

...

<immediate_task_state>
[State of the immediate task at hand]
</immediate_task_satte>

<environment>
- [Key setup/config points]
- ...more...
</environment>

<completed_tasks>
- [Task]: [Brief outcome]
- ...more...
</completed_tasks>

<active_issues>
- [Issue]: [Status/Next steps]
- ...more...
</active_issues>

<code_state>

<file>
[filepath]

**Summary:**
[What this code file does]

**Key elements:**
- [Important functions/classes]
- ...more...
</filepath>

<file>
[filename]
...Similar as above...
</file>

...more files...
</code_state>

<important_context>
- [Any crucial information not covered above]
- ...more...
</important_context>
