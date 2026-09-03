# tests

**You do not need to know anything about rocks, flow or chemistry to run these.**
They are software tests. They ask whether the code does what it says, not whether the
science is right.

---

## What to do

Two commands. That is the whole job.

```bash
cd tests
python run_all_tests.py --fast      # about 15 minutes, a first look
python run_all_tests.py             # about half an hour, the real one
```

Then send back **`tests/test_report.txt`**. It is written automatically and it holds
everything anyone will ask you about.

If you want to know what will run before running it:

```bash
python run_all_tests.py --list
```

## What you should see

A table, one line per group, each ending in one of three words.

| | |
|---|---|
| **ok** | that group passed |
| **FAILED** | something is wrong. The full output is printed underneath and saved in the report |
| **SKIPPED** | the group could not run at all, usually because an optional library is missing. **A skip is not a pass**, which is why it is printed differently |

At the end it says either `Everything that ran, passed.` or `N of M groups FAILED.`
The exit code is 0 when everything passed and 1 otherwise, so it can go in a script.

**Nothing here changes the project.** Every file these tests write goes into a
temporary folder and is deleted at the end. No dataset is built, no model is trained,
and nothing is simulated for longer than a few seconds.

## If something fails

Send `test_report.txt`. Do not try to fix anything. The report already contains:

- which group failed
- its complete output, including the traceback
- the version of Python and of every library you have

That last one is usually the answer. Most failures on a new machine are a missing or
mismatched library, not a bug.

Two failures are expected on some machines and are **not** your problem:

- **tkinter not installed.** The desktop window needs it; nothing else does. The tests
  that need a screen say SKIPPED and the rest run normally.
- **a timeout on a slow laptop.** The two simulator groups are the heavy ones. If they
  time out, say so in your message and run `--fast` instead.

---

## What each file tests

Each one runs on its own, if you want to look at a single group:

```bash
python test_02_units.py         # or any of the others
```

### `test_01_smoke.py` — does everything start at all?

A **smoke test** is the cheapest possible question, asked of everything: does it turn
on? It proves nothing about whether the answers are right. It catches the faults that
would make every other test meaningless.

It checks four things:

1. **every Python file compiles.** A syntax error anywhere.
2. **every library imports.** An import that no longer exists.
3. **every script answers `--help`.** 26 scripts. Each must print its usage and exit
   cleanly, having done nothing else. This catches a script that crashes before it even
   reads its arguments.
4. **the desktop window builds**, with the display faked, and offers the actions it is
   supposed to.

It also times the imports. A library that did real work when you merely imported it
would make every other test slow and unpredictable, and would start a simulation the
moment somebody opened a notebook.

### `test_02_units.py` — one function at a time, against an answer worked out by hand

A **unit test** checks one function, in isolation, on an input small enough that a
person can work out the right answer themselves.

The rule that makes these worth having: **the expected answer never comes from this
code.** Every number in the file was derived from the definition of the quantity, or is
a property that must hold whatever the implementation. A test whose expected value was
produced by running the thing it tests proves only that the code is deterministic.

Some examples of what that looks like here, and you can check each of these on paper:

- Two straight channels side by side, one five voxels wide and one one voxel wide. The
  "how wide is the pore here" map must read **3** in the wide one and **1** in the
  narrow one, because those are the radii that fit.
- A wide chamber reachable only through a one-voxel throat. The local width map must
  read 3 in the chamber. The "what can the inlet deliver here" map must read **1**,
  because nothing wider than the throat can get in. **That single difference is what
  the whole flow capability is for**, so it gets its own test.
- A straight open channel. The pressure must fall in a **straight line** from inlet to
  outlet, and the sideways component of its gradient must be **exactly zero**, because
  nothing pushes the fluid across a straight channel.
- A flow that is the same everywhere conserves mass, so the mass-conservation penalty
  must be **0**. A flow speeding up steadily does not, so the same penalty must be
  **1**. Both are checked: a test that only checked the zero case would pass on a
  function that always returned zero.
- `24` halves to 12, 6, 3, and then stops. So the network's "how many times can this
  grid be halved" function must return **4** for a grid of 24.
- A perfect prediction must cost exactly **0**.

It also checks the table that decides what the network is fed, which is one pure
function and the single most expensive thing in the project to get wrong.

### `test_03_data.py` — files written by this project can be read back correctly

A **round trip test** writes a file, reads it back, and checks that what came out is
what went in.

That sounds trivial. It is not, because the three faults it catches are all silent:

1. a value that changes on the way through, because it was stored in a type too small
   to hold it
2. a key written under one name and read under another, so a field is quietly missing
   and treated as absent
3. a scaling applied on the way in and not undone on the way out, so every number is
   off by a constant nobody notices

None of those raise an error. All of them ruin a result.

It also checks the **layout** every part of this project assumes: that every run points
at a rock that exists, that the table of numbers has one name per column, and that the
real reader can open a file built here by hand.

The last part tests the one operation that changes an existing file in place. It checks
that the three new arrays appear, that nothing already in the file was touched, that
the scaling constants were stored beside them, that a second run is **refused** rather
than silently overwriting, and that the dry-run option writes nothing at all.

### `test_04_repeatable.py` — the same input gives the same answer, twice

A result you cannot reproduce is not a result. If the same code on the same input gives
two answers, then any comparison between two runs is measuring the noise as much as the
change, and nobody can tell which.

Randomness here is not a bug: geometries are generated randomly and training samples
voxels randomly. The rule is that all of it goes through a **seed**. Same seed, same
answer. Different seed, different answer.

**Both halves are tested**, and the second matters as much as the first: a function
that ignored the seed entirely would pass "same seed gives the same answer" perfectly.

It also checks that the scaling can be undone: apply it, reverse it, and you must get
the original numbers back.

### `test_05_refusals.py` — when something is wrong, does it say so, or carry on?

The worst kind of failure is not a crash. It is a run that finishes, writes a file,
prints a number, and is wrong. Nobody looks for a fault they were not told about.

So this project is written to **refuse rather than guess**, and every refusal is
supposed to name the thing to fix. This file does the wrong thing on purpose, twelve
different ways, and checks two things each time: that it failed rather than carrying
on, and that the message says what to do about it.

**Everything in this file is meant to fail.** Among them:

- asking for a flow field the file does not contain. The error must name the command
  that produces it.
- asking for two options that contradict each other. The program must refuse rather
  than picking one silently, because whichever won would be a coin toss the log did not
  record.
- a flag that does not exist, a required argument left out, a file that is not there.
- a model checkpoint stripped of everything except its weights. Nothing in it says how
  its inputs were scaled, so running it would produce a plausible answer that is wrong
  by a constant.
- a rock with no pore space at all, and a rock with no path through it. Both are
  produced by an ordinary sweep, and neither may crash.

A guard that has quietly stopped firing looks exactly like a guard that is working,
which is why they have to be tested by breaking things.

---

## The project's own checks, also run

`run_all_tests.py` also runs the tests that live beside the code they test.

| group | what it proves |
|---|---|
| the window's buttons | every button in the desktop application emits a command line that its script's **real** argument parser accepts. 50 commands |
| the flow pipeline panel | the five steps of the new panel line up with each other, and a failed step stops the sequence instead of running the next one against nothing |
| the sweep mode boxes | both halves of every "a range, or exact numbers" box send the right flags and neither leaks the other's |
| the comments | every source file carries its explanatory header, its section markers and its line comments |
| the flow descriptors | the three pore size maps, 15 checks |
| the pressure solve | the pressure calculation, 14 checks |
| the flow coordinates | the travel-time calculation |
| your own settings | the units and conversions, and that a settings file written by this project can be read back by it |
| the 2D simulator | the simulator's own self-check |
| the 3D simulator | the same, in three dimensions |
| both flow solvers | the two simulators against an answer that comes from neither of them: the exact textbook solution for flow between two flat plates. 28 checks |
| the velocity pipeline | the whole new capability end to end, on a dataset it builds for itself. 33 checks |

---

## What these tests do NOT tell you

Worth being clear, so nothing is over-claimed on your behalf.

**They say nothing about whether the physics is right.** Every test here is about the
software: does it start, does it compute what it claims, does it write files it can
read back, does it refuse bad input. Whether the simulated flow matches a real rock is
a different question, answered by comparing against a reference simulator, and that
comparison has not been done yet.

**They say nothing about whether the model is accurate.** Where a test trains
something, it trains for two epochs on a handful of tiny synthetic rocks. That proves
the shapes line up and the pieces can talk to each other. It proves nothing about the
predictions.

Both of those limits are stated inside the test files themselves, in the same words.

---

## Adding a test

If you find something these miss, the pattern is:

```python
def test_a_sentence_saying_what_must_be_true(self):
    """Why it must be true, and what it would mean if it were not."""
    ...
    self.assertEqual(got, expected, "what the reader should conclude if this fails")
```

Three rules the existing ones follow:

1. **The expected answer must not come from this code.** Work it out, or use a property
   that must hold whatever the implementation.
2. **Test the negative too.** "Same seed, same answer" is passed by a function that
   returns a constant. Add "different seed, different answer".
3. **Say what a failure means.** The message is read by somebody who does not have the
   file open.

Then add it to `GROUPS` at the top of `run_all_tests.py`.
