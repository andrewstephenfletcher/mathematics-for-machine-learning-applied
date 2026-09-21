---
description: Get a graded hint for an exercise, without the solution
argument-hint: <exercise, e.g. 3.8 ex2 or a function name> [level 1|2|3]
---
Give me a hint for: $ARGUMENTS

Follow the hint ladder in CLAUDE.md. Climb no higher than asked, and stop at the lowest rung that
unblocks me:

- **Level 1** — name the relevant concept, definition or book equation number. Nothing more.
- **Level 2** — describe the approach in words: what to build first, what the shape of the answer
  is, what property to exploit.
- **Level 3** — pseudocode, with the key step left for me to write.

If the level is not specified in the arguments, **default to level 1**, and offer the next level
at the end. If it is not clear which exercise I mean, ask before hinting.

Before hinting, read my current attempt if there is one — a hint that ignores the code I have
already written is useless. If my attempt has a specific bug, say which line is suspect and what
invariant it violates, rather than restating the method.

Do not write the solution, and do not edit my files.
