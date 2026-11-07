# Brief: Smarthack XI 2026, PPC theme "24h to Power PPC"

Sources (originals): `team/sources/Smarthack_XI_2026_Tema_PPC.pdf` (the theme, in Romanian),
`team/sources/challenge_GIS.docx`, `team/sources/challenge_OM.docx` (example PPC challenges).
This file is a faithful English summary. If something here and the PDF disagree, the PDF wins.

## The task in one sentence

Build an **AI agent that, based on an ML model trained by us, makes decisions and acts on a real energy
problem at PPC**. At the start of the hackathon PPC presents its business areas and where its hardest
problems are, and each team **chooses which PPC challenge to solve** after that presentation.

## Context: where PPC works and what data exists

PPC covers the whole energy chain: renewable generation, electricity distribution, supply to households and
companies, e-mobility (EV charging) and energy services. Typical data:
- **Time series:** smart-meter consumption, plant / renewable-park production, SCADA from turbines and inverters,
  energy market prices.
- **External:** weather forecasts and history, calendar and holidays, local events.
- **Operational:** grid alarms, intervention and fault history, field reports, EV charging sessions.
- **Customers:** contracts, invoices, payment history, requests and complaints written in free text.
- **Images and spatial data:** drone photos and thermal images, grid maps, charging-station locations.

Daily decisions: which alarm is real, which equipment is at risk of failing, which customers need checking,
which requests are handled first. Technically these reduce to: **time-series regression and forecasting,
anomaly detection with few labels, classification on imbalanced data, clustering and ranking under capacity
constraints.**

"A prediction alone changes nothing": an anomaly score of 0.87 doesn't tell a technician where to go tomorrow
morning. The value appears when the **prediction becomes a decision and the decision becomes an action**.

**What "agent" means here:** a system that **perceives** its environment through data, **estimates** its state with
one or more ML models trained by the team, **decides** based on those estimates and real constraints, and **acts**
on the environment. It can chain models (e.g. classify a customer request by topic, then estimate its urgency
to decide what's handled first) and chain actions (priority list → alert the right team → schedule the
intervention → report the result). It runs **automatically from input data to final action**, and **every
decision is traceable to the model prediction that produced it**.

## Mandatory requirements

1. **At least one ML model trained or fine-tuned by the team**, evaluated on a **separate test set** and compared
   with a **simple baseline**. The training code is in the repo, and the model **runs in our code, not through
   an external API**.
2. **Using an LLM is not enough.** An LLM may coordinate the agent or power the interface, but the agent must
   rely on our model.
3. **The agent must execute actions**, not just converse: prioritize, recommend, generate, alert or plan.
4. **Only public, synthetic, or team-collected and annotated data**, with sources cited.
5. **Code in a public GitHub repository, with commits made during the 24 hours.**

## What a complete project means (they encourage ambition: a full project, not a demo)

Every step from raw data to final action, running automatically with no manual steps in between:
- **Data prepared by the team:** sources chosen and justified; data cleaned, combined, annotated, or realistically
  synthesized; features built for a clear reason.
- **ML model trained and evaluated correctly:** separate test set, comparison with the baseline, **error analysis**
  (where the model is wrong, on which kinds of cases, and what we tried to fix it).
- **A justified decision layer:** predictions become decisions through thresholds chosen for a reason, **real
  constraints** (e.g. how many cases a team can check per day) and the model's confidence.
- **Concrete actions:** something usable, such as a priority list, an alert, a report, a scheduled intervention.
- **Handling hard situations:** missing data, rare cases, wrong inputs and model uncertainty. A good agent can say
  "I'm not sure" and **hand the case to a human**.
- **Explainability:** the agent shows why it decided something (feature importance, similar historical cases).
- **Reproducibility:** anyone on the jury can run the project from the README, and the presentation's results
  can be regenerated with our evaluation script.

## Judging (100 points). Winner: the best idea for PPC *and* the best implementation; one without the other isn't enough.

| Criterion | Points | What the jury looks for |
|---|---|---|
| Idea and relevance for PPC | 25 | How real and important the problem is, how well it's framed, how credible it would be in PPC colleagues' daily work |
| ML quality | 25 | Data preparation, model choice, correct evaluation on a separate test set, comparison with baseline, error analysis |
| Robustness | 10 | Missing data, rare cases, wrong inputs, handling uncertainty |
| The agent | 25 | A complete automatic flow: data → model → decision → action, with a justified decision layer |
| Explainability | 10 | The agent shows why it made a decision |
| Demo and repository | 5 | Clear presentation, reproducible README, clean code, commit history |

## Deliverables

- The working project in a **public GitHub repository**, together with the presentation materials (slides,
  notebook, etc.).
- A **live demo**, followed by technical questions from the jury. We choose the presentation format.

## Example PPC challenges (from the .docx files)

### A. GIS: a digital model of the grid that updates itself as the network evolves
- **Area:** GIS / network technical data. Mentor: Ionuț-Valentin Isopescu.
- **Context:** the quality and completeness of the high-, medium- and low-voltage network model in GIS, the base
  layer for digitalizing and automating outage management, investment planning and grid connections.
- **Users:** operations, maintenance, investment planning, connections; indirectly the customers affected by
  decisions made on incomplete data.
- **Problem:** the network changes constantly (new works, connections, reconfigurations), but the GIS model lags
  behind: missing or mispositioned sections, field changes that never get back into the system, stale data.
  The gap is invisible until something depends on it (a delayed intervention, a badly grounded investment).
  Fixing data point by point is expensive, and without continuous updating any correction degrades from day one.
- **Constraints:** must work alongside the existing distribution GIS platform and respect distribution regulations.
- **They offer:** a sample of GIS data, regulatory guidance, mentoring from the model's team.
- **Success:** a working mechanism that **identifies field changes for updating the model**, with a **visible,
  clearly communicated confidence level for network sections not recently verified**.

### B. O&M: one app showing energy produced, consumed and stored at the customer (PV + BESS)
- **Area:** energy / prosumers & O&M. Mentor: Laurențiu.
- **Context:** more residential and industrial customers install solar panels (PV) and batteries (BESS). PPC wants a
  solution that manages energy at the customer level and is also usable by O&M (operations and maintenance).
- **Users:** customers with PV and/or BESS; internally the O&M teams looking after these installations.
- **Problem:** customers see production and consumption in pieces, in each device's own app, and O&M finds out
  about problems only after a loss or fault. Data sits separately in each device and is never combined, so the
  customer doesn't know when to consume to pay less, and O&M can't catch early that something is off
  (underperforming production, poorly used storage, a fault starting).
- **Constraints:** must work alongside existing equipment and apps (inverters, energy management systems) without
  replacing them; must respect prosumer and data-protection rules.
- **They offer:** guidance on prosumer and data-protection rules, possibly a test environment, sample data from real
  installations, support from the O&M team.
- **Success:** a single view of produced / consumed / stored energy that helps the customer consume smarter (pay
  less) and helps O&M catch abnormal operation early, with a **visible reduction in wasted energy or in
  time-to-detect a problem**.

## What this means for us (team notes, not from the organizers)

- The final challenge is chosen **after PPC's presentation at the start**. Brainstorm ideas for each area now, so we
  can commit quickly once we hear the presentation and talk to mentors.
- 50 of 100 points are "idea for PPC" + "agent": the data → model → decision → action loop matters as much as the model.
- Shape of a winning repo: `data/` (sources + synthesis scripts), `train.py` (runs on the lab's GPUs via
  `lab submit <card> --run train`), `eval.py` (separate test set, baseline, error analysis; this regenerates the
  presentation's numbers), the agent (decision layer + actions + "I'm not sure → human"), and an explanation
  view in the web app.
- The repo must be **public** by submission, with commits throughout the 24 hours. The lab gives us that naturally.
