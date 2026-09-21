---
description: Generate exercises for a section of Mathematics for Machine Learning
argument-hint: <section e.g. 3.8> [optional focus or notes]
---
Generate the exercise notebook for MML section $ARGUMENTS.

1. Read CLAUDE.md, the chapter README and the ROADMAP.md entry for this chapter.
2. Read this section of the PDF in book/ to align notation, scope and equation numbers. The
   chapter README lists each section's page range (PDF page = printed page + 6). If the PDF is
   missing, say so and flag that equation numbers may need checking.
3. Look at what already exists in this chapter and in mml/ so exercises build on earlier work
   rather than repeating it.
4. Propose a short plan — learning objectives, and a list of exercises across the Core /
   Intuition / Safety tiers with estimated time — and WAIT for my approval before writing files.
5. After approval, create the notebook (with its jupytext .py twin), any stubs in src/, and tests
   in tests/. Stubs must raise NotImplementedError with a docstring giving inputs, outputs, shapes
   and the book equation number. Do not write solutions.
6. Run the tests to confirm they fail cleanly on the stubs — NotImplementedError, not import or
   syntax errors — and run ruff.
7. Update the chapter README checklist and ROADMAP.md.
