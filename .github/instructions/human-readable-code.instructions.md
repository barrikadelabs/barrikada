---
description: "Use when writing, refactoring, or reviewing code. Enforces clean human-readable style, minimal necessary typing, and subagent-first task decomposition for complex work."
name: "Human Readable Code And Orchestration"
---
# Human Readable Code and Orchestration

- Write code that feels natural and maintainable, as if authored by a careful human engineer.
- Prioritize clarity over cleverness: meaningful names, simple control flow, and small focused functions.
- Keep type annotations practical and lightweight.
- Avoid extensive or deeply nested typing when it does not improve safety or readability.
- Add comments only when intent is not obvious from the code itself.
- For non-trivial tasks, decompose work and delegate exploration or isolated subtasks to subagents.
- Keep the main agent focused on orchestration: plan, integrate results, and verify outcomes.
- Prefer incremental, reviewable changes rather than large monolithic rewrites.
