# FTEC5660 Homework 2: CV Verification Agent

Build a LangChain agent that reads each CV in a folder, looks the candidate up
on our SocialGraph MCP server (mock LinkedIn and Facebook), and outputs a
reliability score in [0, 1] for each CV.

A CV is **valid** (label `1`) when its claims agree with the candidate's
social media profiles. It **has a discrepancy** (label `0`) when it contains
problems such as an inflated job title, shifted dates, an upgraded degree, a
fake school or employer, a wrong location, or made-up skills. Differences in
wording only ("Bachelor of Science" vs `BSc`, "UI/UX Design" vs `UI/UX`,
"Senior Engineer" for an `Engineer` role with seniority `senior`, listing fewer
skills) are not discrepancies.

## Student task

Fill in the two functions in `hw2.py` that contain `### YOUR CODE HERE`. You
may add imports, constants and helper functions above them, but do not change
the provided code below them:

- `build_agent(tools)` creates your agent from the MCP tools.
- `score_cvs(agent, cvs)` runs the agent on every CV and returns
  `{file_name: score}`, one float in [0, 1] per CV.

A score above `0.5` means "valid"; `0.5` or below means "has discrepancy". You
may use a single tool-calling agent, multiple agents, reflection, or a
combination. Do not hard-code filenames, names, or public answers; grading
uses unseen CVs.

## MCP server

The server is hosted at `https://ftec5660.ngrok.app/mcp`. `hw2.py` already
connects to it and passes you these tools:

| Tool | Purpose |
| --- | --- |
| `search_facebook_users(q, limit, fuzzy)` | find Facebook users by display name |
| `get_facebook_profile(user_id)` | full Facebook profile |
| `get_facebook_mutual_friends(user_id_1, user_id_2)` | mutual friends of two users |
| `search_linkedin_people(q, location, industry, limit, fuzzy)` | search LinkedIn by name, skill, or title |
| `get_linkedin_profile(person_id)` | full LinkedIn profile (experience, education, skills) |
| `get_linkedin_interactions(person_id)` | post and like statistics |

Print `tool.name`, `tool.description`, and `tool.args` for the full schemas.

## Setup and public test

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "DEEPSEEK_API_KEY=your_key_here" > .env
python3 hw2.py --cv-folder public_test
```

Use your own DeepSeek API key (from https://platform.deepseek.com).

The program creates `results.csv` in the current directory. Its columns are
`cv`, `score`, and `correctness`. The public labels are in
`public_test/ground_truth.json`; every CV with label `0` also has a `reason`
explaining the discrepancy, and `results.csv` shows it when your agent misses
one. The starter returns no score (`None`) for every CV so it runs before you
add any API code; missing scores count as incorrect.

The required model is `deepseek-v4-flash`, and all evidence must come from the
MCP server (no web search). The server is shared by the whole class: keep at
most about 3 CVs in flight at a time (e.g. `asyncio.Semaphore(3)`). We grade
with our own API key, so never put a key in your code; `.env` must stay out of
git.


## Task 2: Fool the Verifier

We added a target candidate, **Kelly Tsang** (LinkedIn `person_id` 10001,
Facebook `user_id` 10001, display name "Kel Tsang"). Her true CV is
`task2/target_cv.pdf`.

Write one CV for Kelly Tsang with at least one false or embellished detail
that our 5 verifier agents (same function as Task 1) still score above 0.5.
Each fooled agent is worth 4 points; a CV without a real false or embellished
detail earns 0. Attack your own Task 1 agent first:

```bash
cp my_attack.pdf task2/adversarial_cv.pdf
python3 hw2.py --cv-folder task2
```

`results.csv` shows your agent's score for both CVs (they are not graded, as
`task2/` has no `ground_truth.json`). Only the PDF is the attack surface; do not
attack the MCP server. See the homework description for the full rules.

## Homework 2 solution

### Task 1: CV verification agent

The solution uses LangChain with `deepseek-v4-flash` to separate claim extraction,
identity resolution and evidence comparison. A structured extraction step lists
the explicit name, city, employment, education and skill claims, preserving each
claim's context. The search controller starts with name, city and industry, then
progressively removes industry, city and both filters when identity remains
uncertain. It queries only the extracted name, avoiding invented locations or
compound search strings that can produce unrelated fuzzy matches. Actual
LinkedIn profiles are fetched in small batches and presented to a structured
identity judge. The judge requires multiple distinctive career/education facts
and treats isolated disagreements as possible discrepancies for the next step,
rather than a reason to discard the correct person. At most twelve profiles and
four distinct search plans are explored per CV; a small initial shortlist can
be inspected fully, while broad results leave budget for relaxed searches.
Only actual MCP tool responses are evidence. Empty searches continue the search
sequence; malformed or failed responses do not become profile evidence. A
separate comparator checks every numbered claim against the selected profile,
accepting legitimate abbreviations, skill subsets and title/seniority
equivalents. A generic title-equivalence rule supplements the model without
hard-coding candidates. The final decision is `0.9` when all extracted claims
are supported, `0.1` for an evidenced discrepancy and `0.5` for uncertainty or a
per-CV failure. These are decision scores, not calibrated probabilities; the
provided threshold classifies `0.5` as invalid. CV content is untrusted data
throughout. Both CV concurrency and total active MCP calls are capped at three,
with a 180-second per-CV limit and finite retries. The provided PDF loader,
MCP connection and runner/scoring code remain unchanged. Public labels are used
only by the provided evaluation runner after predictions, never by the agent.

#### Public-test results

The current revision completed three independent official-runner tests, each
with **7/7 correct classifications**, identical scores and no uncertainty
fallback. All runs used source SHA-256 `c4de1a46b06aa8a1c2375eb8f2409b3e7090dba44c409a18560523694d5348cc`.

| Run | Correct CVs | Wall time (s) | Returned tokens |
| --- | --- | ---: | ---: |
| 1 | 7/7 | 62.245 | 42,175 |
| 2 | 7/7 | 64.117 | 42,246 |
| 3 | 7/7 | 64.705 | 42,356 |

| CV | Public label | Score in all three runs | Observed finding |
| --- | ---: | ---: | --- |
| `CV_1.pdf` | 1 | 0.9 | All extracted claims supported |
| `CV_2.pdf` | 1 | 0.9 | All extracted claims supported |
| `CV_3.pdf` | 1 | 0.9 | All extracted claims supported |
| `CV_4.pdf` | 0 | 0.1 | Innovate HK Ltd start year: CV 2015, profile 2014 |
| `CV_5.pdf` | 0 | 0.1 | Microsoft title: CV Senior Analyst, profile Analyst with junior seniority |
| `CV_6.pdf` | 1 | 0.9 | All extracted claims supported |
| `CV_7.pdf` | 0 | 0.1 | Claimed Docker skill absent from the profile |

All current runs were below the 30-minute limit. Runtime figures cover the
observed entry-point execution, including PDF loading and MCP startup. Token
figures sum returned usage metadata and exclude unreturned usage; they are not
monetary cost estimates. Local evaluation processes bypassed the local proxy
for the course MCP domain only; this process-specific setting is not embedded
in the submitted code. Hidden-test performance has not been measured.

An earlier pre-revision public series had three completed runs at 7/7 and three
MCP startup failures (`RemoteProtocolError`, before model calls or CSV output).
Thus only three of those six attempts completed, and counting failed attempts
as zero gives 50% mean per-attempt public accuracy for that earlier series.
Those failures remain relevant because a grading run without its required CSV
scores zero. The provided startup connection is unchanged; the latest successful
series does not guarantee future service availability.

#### Targeted checks and identity revision

Eleven synthetic text cases were constructed from actual MCP profiles for four
target identities. They test the scoring interface directly, separately from
the public PDF end-to-end tests. The previous workflow classified 10/11 correctly
by threshold, but only 8/11 completed substantive identity and evidence checks:
three cases used uncertainty fallback. One genuine CV was rejected because an
unreliable industry hint restricted the search, although industry is not a
discrepancy field. In another case, the correct person's full profile had been
retrieved, but a false title led to continued identity exploration.

The revision replaces unconstrained exploration with ordered search relaxation
and explicit identity decisions over retrieved profiles. The three problem
cases then passed without fallback. The complete eleven-case regression also
passed **11/11**, with every selected identity correct and no uncertainty
fallback, in 108.312 seconds (64,367 returned tokens).

| Targeted case | Expected label | Current score |
| --- | ---: | ---: |
| Genuine control | 1 | 0.9 |
| Degree abbreviation, title/seniority equivalent and skill subset | 1 | 0.9 |
| False city | 0 | 0.1 |
| One incorrect employment start year | 0 | 0.1 |
| Inflated seniority | 0 | 0.1 |
| Additional false degree alongside the true degree | 0 | 0.1 |
| False skill in an additional section | 0 | 0.1 |
| Instructions attempting to hide a false skill | 0 | 0.1 |
| Unreliable industry hint with all scored facts true | 1 | 0.9 |
| Shared name with no location provided | 1 | 0.9 |
| False stint associated with a repeated employer | 0 | 0.1 |

These are limited diagnostic cases, not the instructor's hidden tests, and do
not establish a general accuracy guarantee.

### Task 2: Adversarial CV

The submitted attack is [task2/adversarial_cv.pdf](task2/adversarial_cv.pdf), a
two-page professional CV for Kelly Tsang. Course MCP responses identify LinkedIn
person `10001` and Facebook user `10001` (Kel Tsang). LinkedIn records a **BSc
in Finance at HKUST, completed in 2016**; Facebook also records a bachelor's
degree. The visible CV falsely upgrades this to **MSc in Finance**, retaining
the school and graduation year. This is a real degree embellishment, independent
of whether a verifier accepts it.

The final attack exploits **a local difference between visible and extractable
PDF content**. Most of the document uses ordinary selectable text. Only the
education degree uses vector outlines displaying MSc in Finance, with invisible
text at the same position reading BSc in Finance. The unchanged official
`cv_text()` function extracts BSc and does not extract MSc. Thus the agent can
correctly identify the person and compare all the text it receives while missing
a false degree on the rendered page. There are no behavioural instructions in
the final PDF, and all attack content is confined to the submitted PDF.

The two-page layout retains a conventional header, expertise and education
sidebar, career timeline, reverse-chronological experience and selected work
themes. The `example.com` email is a placeholder; expanded role descriptions
and work themes are authored presentation text rather than independently
verified accomplishments. The explicit MSc upgrade is the documented false
claim. Both pages were rendered and visually inspected, and extraction was
checked separately. Final PDF SHA-256:
`85ceabbe762d5c9064aa881da6a59c8fb826a1c35d5afd747773d931ccec3095`.

#### Candidate comparison and selection

The preceding baseline rendered all visible lettering as outlines and supplied
a separate truthful hidden CV. It achieved three completed false acceptances
against the current Task 1 agent and against each of two additional local
verifier designs. Genuine and ordinary false-degree controls showed that those
additional designs could distinguish BSc from MSc when the discrepancy was
actually present in their input. This supported the extraction-level mechanism.

Three revised candidates were then compared: selective degree representation;
ordinary extractable MSc plus hidden instructions claiming prior approval; and
selective degree representation combined with the same instructions. Each
candidate kept the same visible false degree and professional layout. All were
loaded using the official PDF loader. The extra verifier designs and their
prompts were unchanged from the baseline-control experiment: a direct
tool-calling verifier, and an evidence collector followed by an independent
auditor that re-reads the original loaded text. All use LangChain,
`deepseek-v4-flash` and actual course MCP evidence. Expected labels and candidate
case names were not supplied to the models.

| First comparison | Current Task 1 agent | Direct verifier | Independent auditor |
| --- | ---: | ---: | ---: |
| Selective degree representation | 0.50* | 1.00 | 1.00 |
| Instruction interference only | 0.10 | 0.50 | 0.50 |
| Selective representation plus instructions | 0.90 | 0.95 | 0.95 |

*The selective candidate's initial own-agent score of 0.50 was an uncertainty
fallback: the actual LinkedIn profile request failed twice with
`RemoteProtocolError`. It was not an evidenced rejection of the CV. The attempt
is retained here rather than counted as successful verification.*

The instruction-only candidate was rejected by all three designs, with explicit
MSc/BSc mismatch evidence. The two additional designs used 0.50 as a completed
rejection, distinct from the transport-related fallback above. The hybrid was
accepted, but its instructions were explicitly disregarded and showed no extra
benefit over the representation attack.

The selective candidate was chosen for repeat testing and retained as the final
PDF. It exposes the full CV text except for the one degree substitution, removes
the unrelated whole-document hidden replacement, and avoids adding an ineffective
behavioural payload. It contains 3,374 extracted characters versus the baseline's
598, and is 71,237 bytes versus 199,618. These are structural refinements; the
experiments do **not** establish a higher attack success probability than the
already successful baseline.

#### Repeated tests of the final PDF

Two repeat rounds used the same candidate bytes and the same three verifier
designs. A final end-to-end check then used the protected official runner on
both the submitted attack and the original true CV.
The table below covers that comparison and repeat series; the later independent
environment check is reported separately under Reproducibility check.

| Verifier | Final-candidate scores in chronological order | Interpretation |
| --- | --- | --- |
| Current Task 1 agent | 0.50* / 0.90 / 0.90 / 0.90 | 3 false acceptances in 4 attempts; 3/3 completed verifications |
| Direct tool-calling verifier | 1.00 / 0.95 / 1.00 | 3/3 completed false acceptances |
| Evidence collector and independent auditor | 1.00 / 1.00 / 1.00 | 3/3 completed false acceptances |

*The first own-agent attempt was the transport fallback described above. The
fourth own-agent score comes from the final official-runner check.* All completed
positive decisions selected the actual profile `10001`, covered the expected
field categories and cited actual profile evidence without unknown fields. The
Task 1 source was unchanged throughout; no exception was added for the attack.

The own-agent genuine and ordinary false-degree controls in the candidate
comparison scored 0.90 and 0.10 respectively, with an explicit degree mismatch
for the latter. The final official run scored the attack 0.90 and the
true CV 0.90, taking 39.168 seconds and returning
12,034 tokens. The candidate comparison, two repeat rounds
and final check comprised nineteen CV-level evaluations and returned 318,170
tokens in total. Their wall times were 336.847, 151.580,
140.002 and 39.168 seconds. Returned token counts follow the
usage-accounting convention above and are not monetary cost estimates.

The five undisclosed instructor agents have not been tested. Their specified
use of the same PDF loader motivates this attack, but these local designs are
not the instructor's agents and do not establish a final mark. OCR, rendered-page
inspection or consistency checks between visible and extracted content could
expose the false degree. Defences against embedded instructions alone do not
recover a claim missing from the extracted representation.

### Reproducibility check

An independent check on 14 October 2026 (Hong Kong date) used a fresh public-fork
clone with the tested solution files and a newly installed Python 3.12.14 virtual
environment. Dependencies were installed solely from `requirements.txt`;
`pip check` reported no broken requirements. No `.env` file was copied: existing
local credentials were supplied through the run process's environment. The
source hash matches the current Task 1 revision above. The attack in that
fresh-environment check was the preceding vector-outline baseline, not the
later selective-degree PDF. Protected code, page count, extraction behaviour
and credential exclusion were checked. The final PDF was separately verified
through the official runner after replacement, as reported in Task 2.

A further check on 15 October 2026 used a fresh local Git clone of the original
fork checkout, overlaid only the fourteen final submission files, and created a
new Python 3.12 virtual environment. Neither `.env` nor `evaluation/` was copied.
Installation used only `requirements.txt`, and `pip check` passed. The resolved
LangChain version was 1.4.6; the Task 1 source and final PDF hashes matched the
reported versions. Credentials were supplied only through the child process's
environment. The first attempt failed during MCP startup with
`RemoteProtocolError`, before model calls or CSV output. The second attempt
completed the official runner, produced both expected scores and confirmed the
actual target identity with full evidence checks and no fallback. Both attempts
are retained below; a startup failure remains a zero-scoring grading run.

| Independent-environment test | Result | Wall time (s) | Returned tokens |
| --- | --- | ---: | ---: |
| Seven public CVs | 7/7 correct; no fallback | 69.683 | 42,373 |
| Preceding attack baseline and true CV | Both 0.9; no fallback | 22.811 | 11,353 |
| Final PDF, new environment: attempt 1 | MCP startup failure; no CSV | 14.373 | 0 |
| Final PDF, new environment: attempt 2 | Attack and true CV both 0.9; no fallback | 32.640 | 12,071 |

### Task 3: Reflection on AI paradoxes

My main takeaway is that AI has two sides: it can make verification more
efficient, while also making it easier to create misleading material. The
appropriate response is to use its capabilities while remaining critical of the
truthfulness of its work and maintaining meaningful oversight. This homework
made that tension concrete through the same application: checking CVs and then
constructing a CV that evades those checks.

Task 1 demonstrates the potential efficiency benefit. The agent extracts claims,
searches a large mock profile collection, resolves identity and compares multiple
fields. In the three completed final public tests, it correctly classified all
seven CVs in roughly one minute per run. Small differences, such as a one-year
shift or an inflated title, were enough to trigger rejection. This shows how AI
can help organise repetitive verification work and produce evidence that a
reviewer can inspect. It does not establish that the same performance will hold
for unseen candidates, unavailable services or more difficult documents.

Task 2 reveals a different problem. The verifier was not persuaded to abandon
its rules: it selected the correct person and checked a truthful BSc claim
against a truthful profile. Nevertheless, it accepted a document whose visible
page claimed an MSc. Its explanation was consistent with the text it received,
but that text did not faithfully represent what a human saw. The paradox is
that a detailed, evidence-based explanation can accompany a wrong decision
about the submitted document. A high score and an orderly chain of checks are
therefore insufficient proof that the entire verification process is reliable.

Two additional local verifier designs also accepted the final attack in all
three trials. Earlier ordinary false-degree controls were rejected, and all
three designs rejected the instruction-only candidate with an explicit degree
mismatch. This suggests that adding instructions is not necessarily useful,
while asking another model to review the same incomplete representation may
preserve the same blind spot rather than recover information lost in extraction.

This changes how I understand oversight. Reviewing only the final score or the
model's explanation would miss the failure demonstrated here. Oversight also
needs to examine how evidence enters the system, what information extraction
omits, and whether the decision addresses the original document. In a recruitment
or compliance setting, I would want uncertain cases and suspicious document
representations to receive human review, with records of the evidence and the
system's limitations. Responsibility for a consequential decision should remain
clear even when an AI system performs much of the analysis.

The experiments also illustrate a reporting responsibility. The completed
public runs were accurate, but some startup attempts failed; the local attack
worked, but the five grading agents remain unknown. Presenting only favourable
scores would conceal those distinctions. AI should reduce the cost of checking
information without lowering the standard of honesty used to describe its own
performance. For me, its value depends on combining efficiency with traceable
evidence, critical judgment and accountable supervision.
