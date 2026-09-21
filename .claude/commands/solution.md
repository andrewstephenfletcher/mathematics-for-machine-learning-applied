---
description: Write a reference solution for an exercise (only when I ask)
argument-hint: <exercise, e.g. 3.8 ex2 or a function name>
---
Write a reference solution for: $ARGUMENTS

This command is the **only** route by which you write solution code, and it runs only when I
invoke it. If I have not attempted the exercise at all, say so once and ask whether I want a hint
instead — then do as I say.

1. Read the exercise stub, its docstring and its tests so the solution matches the required
   signature and shapes exactly.
2. Write the solution to `chapters/<chapter>/solutions/<same filename as the stub>`, creating
   `solutions/` if needed. **Never overwrite my attempt in `src/` or in the notebook.**
3. Comment the key steps — the ones where the mathematics becomes code: why this factorisation,
   why this axis, where the book's equation appears in the line. Skip commentary on the obvious.
4. Where a more naive version exists, mention it and say why the chosen one is better
   (stability, cost, generality).
5. Run the exercise's tests against the solution to prove it passes, and show the output.
6. End with the one idea I would most likely have got wrong, and a short check I can do to
   confirm I now understand it.
