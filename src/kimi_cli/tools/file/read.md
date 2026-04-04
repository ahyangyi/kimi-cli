Read text content from a file.

**Tips:**
- Make sure you follow the description of each tool parameter.
- A `<system>` tag will be given before the read file content.
- The system will notify you when there is anything wrong when reading the file.
- This tool is a tool that you typically want to use in parallel. Always read multiple files in one response when possible.
- This tool can only read text files. To read images or videos, use other appropriate tools. To list directories, use the Glob tool or `ls` command via the Shell tool. To read other file types, use appropriate commands via the Shell tool.
- If the file doesn't exist or path is invalid, an error will be returned.
- If you want to search for a certain content/pattern, prefer Shell with `rg` (or the Grep tool if Shell is unavailable) over ReadFile.
- Content will be returned with a line number before each line like `cat -n` format.
- Use `line_offset` and `n_lines` parameters when you only need to read a part of the file.- When context budget is tight, prefer reading ≤200 lines at a time with `line_offset` + `n_lines` rather than loading the whole file.
- `line_offset` can be negative to count backward from the end of the file. For example, `line_offset=-200` reads the last 200 lines.
- The tool result message includes the file's total line count when it is known.
- The maximum number of lines that can be read at once is ${MAX_LINES}.
- Any lines longer than ${MAX_LINE_LENGTH} characters will be truncated, ending with "...".
