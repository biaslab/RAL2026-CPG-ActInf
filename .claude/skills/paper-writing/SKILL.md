---
name: paper-writing
description: House style rules for writing and editing the LaTeX manuscript in this repo (main.tex and anything under archive/paper-*/). Load before drafting or revising any prose, abstract, caption, or rebuttal text.
---

# Paper writing style

House rules for prose in `main.tex`. These are the author's preferences, not
generic advice; follow them over your defaults.

## Citing prior work

**Do not write "[Name] et al." in a sentence unless it cannot be avoided.**

Attribute the work to the citation, not to a named person in the running text.
The bibliography already carries the authorship, so naming authors inline spends
words, pulls the reader's attention to who did the work rather than what it
shows, and reads as deference. Write about the method or the finding and let
`\cite{}` do the attribution.

```latex
% avoid
Ajallooeian et al.\ use virtual springs to pull the trunk level \cite{ajallooeian2013central}.
Cully et al.\ let a damaged robot search a precomputed repertoire \cite{cully2015robots}.

% prefer
Virtual springs spanning a trunk-fixed plane and a horizontal reference pull the
trunk level \cite{ajallooeian2013central}.
A damaged robot can search a precomputed behavioral repertoire by trial and
error \cite{cully2015robots}.
```

It *can* be avoided in almost every case. Reach for a rephrasing first:

- lead with the mechanism or result — "Fitting a Gaussian process surrogate to
  the parameter-score pairs seen so far ... \cite{eriksson2019scalable}";
- lead with the artifact — "The RMA policy infers a latent encoding of the
  condition ... \cite{kumar2021rma}";
- lead with the field — "A complementary line tunes gait parameters as online
  black-box optimization \cite{...}".

Legitimately unavoidable cases are narrow, e.g. when the paper is contrasting
two specific groups' positions on the same question and the reader must track
which is which across several sentences, or when a method is universally known
by its authors' names. If you use the form, it should be a considered choice you
can defend, not a default.

The same applies to possessives ("Cully's repertoire") and to "the authors of
\cite{x}", which is the same move with more words.
