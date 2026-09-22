# Paper review artifacts (archived)

Status: ARCHIVED. Workflow artifacts of the paper's R4/R5 revision (July 2026), moved here from
`articles/paper/` on 2026-09-22 (issue #101) so the paper source directory holds only inputs of
the build (`paper.typ`, `appendix.typ`, `results.typ`, `refs.bib`, `data/`, `figures/`, `scripts/`, `fonts/`).
Nothing in the tree reads these files.

- `2026-07-10_reviewer_report_1.md`, `2026-07-10_reviewer_report_2.md`: AI-generated reviewer
  reports (not human peer review) that drove the revision landed in `e1ba688` (full-review fixes:
  claims, stats, domain, bib), `6d0cb10` (second-review fixes: data, figures, legibility) and
  `608667c` (reports committed alongside the response). Report 2's Section 5 request (B, C) is
  what the protocol-matched RL baseline of `articles/paper/CHANGES.md` answers.
- `2026-07-10_review_state.json`: the finding-by-finding state of the multi-dimension review
  workflow (`workflow wf_b38fd20e-6b6 run-1`) behind those reports.

The revision itself is in the git log; these files are kept only so a reader of that history can
see what the reports asked for.
