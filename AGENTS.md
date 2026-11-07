# Working in this repo

This team coordinates through **lab**, a shared board of cards. Humans and agents use the same `lab` CLI.
Your identity is `$LAB_ID` (or `.lab/identity`). Agents never talk to each other directly — only through the lab.

## Every time you start a task

1. Read what's new: `lab inbox` (hooks usually show this automatically at the start of each turn).
2. Look at the board: `lab status`. For optimization/ML work also `lab top` and the notes on recent cards.
3. Read `team/brief.md` (the challenge and how it is judged), `team/plan.md`, `team/contracts/` and `team/decisions.md`.
   They override anything said in chat.

## The card workflow

- Pick up work: `lab take <id>` (creates and checks out branch `card/<id>`). Prefer open cards in your area over inventing new ones.
- New work you won't do yourself, or a follow-up idea: `lab propose "title" --body "..." [--parent <id>] [--paths "web/src/x/**"]`.
  A good card is self-contained: goal, acceptance (a command or observable result), files, contract, what not to touch.
- Commit on the card branch, then `lab submit <id>`. This pushes, then runs the checks (and the preview, or training) on the lab.
  - `merge=auto` cards merge into main by themselves when the check passes. `review` waits for `lab approve`. `never` = experiment.
  - Training / sweeps: `lab submit <id> --run train --cmd "python train.py --lr {lr}" --grid lr=1e-3,3e-4`.
- Results come back to the card and to your inbox. You do not need to wait: work on something else and read results next turn.
- Logs: `lab logs <run> [-f]`. Cancel: `lab cancel <run>`.
- Never push to `main` directly. Never train for more than a few minutes locally — submit to the lab instead.
- `lab sync` merges new main into your branch (hooks do this between turns when it's safe).

## Communicating

- Feedback or conclusions about a card: `lab note <id> "..."`. After an experiment finishes, always write what you learned.
- Questions to someone specific: `lab msg <person|person/agent> "..."`; replies: `lab reply <msg-id> "..."`.
- Things everyone must know (contract change, dropped feature): `lab announce "..."` and record it in `team/decisions.md`.
- Post only to: ask something you can't find in the repo, report a blocker, announce a contract change, or give feedback/conclusions.
  No acknowledgements ("thanks", "got it"). Keep messages under ~5 lines; link to cards, commits or files instead of pasting.
- `--urgent` only when someone must change course right now (e.g. you are about to rewrite a file they are editing).
- Never block waiting for a reply: work against the contracts in `team/contracts/` and adjust when the answer arrives.

## Delegating

- Split big work into cards; others (humans or agents) take them. Don't assign work to a person's agent without a card.
- A card you created that someone else took is theirs; comment with `lab note`, don't edit their branch.
- Don't delegate a delegated card further; if it's unclear, ask its proposer.
