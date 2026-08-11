# Putting PRT-DeepONet on GitHub through the website

No git, no GitHub Desktop, no command line. Everything below happens in your
browser and in Windows File Explorer.

Two limits of the GitHub web uploader decide how this is done:

* one upload can carry at most **100 files**
* one file can be at most **25 MB**

Your project as it stands breaks both, because `2D\Domains` holds 3000 loose
`.dat` files and `3D\.venv` holds tens of thousands. So the first job is to
build a clean copy that contains only what belongs in a repository.

---

## Step 1. Build the clean copy

In File Explorer, go to your Desktop and make a new folder called

    PRT-DeepONet-github

Now copy these into it, keeping the same folder names.

**From the root of PRT-DeepONet, copy these six files:**

    FIXES.md
    HEEWON_DATA.md
    PYTHON_VS_COMPLAB.md
    README.md
    STEPS_2D.md
    check_everything.py

**Copy these four folders whole:**

    bridge
    docs
    examples
    gui

**Make a folder called `3D` inside the copy, and put two folders in it:**

    3D\tools      (copy of your 3D\tools)
    3D\model      (copy of your 3D\model)

**Make a folder called `2D` inside the copy, and put in it:**

    2D\LICENSE
    2D\README.md
    2D\PRT-DeepONet_Irreversible_Sorption.ipynb
    2D\PRT-DeepONet_Monod.ipynb
    2D\PRT-DeepONet_Reversible_sorption.ipynb
    2D\models          (the three _load.ipynb files)
    2D\geometries      (the three .npz files)
    2D\parameters      (the three .pt files)
    2D\Domains\Input_domains.zip      <-- the zip only

That last line matters. `Input_domains.zip` is 2.4 MB and already contains all
3000 domain files. Do **not** copy the loose `.dat` files; they are 61 MB and
would take thirty separate uploads.

## Step 2. Delete what should never be uploaded

Inside `PRT-DeepONet-github`, delete these if any of them came along:

    3D\tools\_to_delete
    any folder called __pycache__
    anything called .venv
    the work folder

None of it belongs in a repository. `.venv` is a Python environment that
another machine cannot use anyway, and `work` holds simulation output that
gets regenerated.

## Step 3. Check for oversized files

Right-click `PRT-DeepONet-github`, choose Properties, and look at the size.
It should be roughly 5 to 10 MB. If it is much larger, something from Step 2
is still in there.

Then check `2D\parameters`. If any `.pt` file is larger than 25 MB the web
uploader will refuse it, and you will need to leave those three files out and
mention in the README where the trained weights can be obtained.

## Step 4. Create the repository on GitHub

1. Go to **github.com** and sign in. If you have no account, click Sign up and
   use your UGA email.
2. Click the **+** in the top right corner, then **New repository**.
3. Repository name: `PRT-DeepONet`
4. Description: `Geometry-aware neural operator for pore-scale reactive transport, with a lattice Boltzmann simulator (PRT-LB) for 2D and 3D.`
5. Choose **Public**.
6. Tick **Add a README file**.
7. Under **Add .gitignore**, choose **Python** from the dropdown.
8. Under **Choose a license**, pick **MIT License** unless your lab requires
   another one.
9. Click **Create repository**.

You now have a repository with three files in it.

## Step 5. Upload the project

1. On the repository page, click **Add file** near the top right, then
   **Upload files**.
2. Open `PRT-DeepONet-github` in File Explorer.
3. Select everything inside it with **Ctrl+A**. Do not select the folder
   itself, select its contents.
4. Drag the selection onto the grey box in the browser that says
   "Drag files here to add them to your repository".
5. Wait until every file shows in the list. It should be about 70 files.
   Folder structure is preserved, so `3D/tools/prtlb_3d.py` lands in the
   right place.
6. In the box at the bottom, where it says "Commit changes", type
   `Initial upload of the 2D and 3D code`.
7. Leave "Commit directly to the main branch" selected.
8. Click **Commit changes**.

If GitHub complains that there are too many files, upload in two goes: first
the `2D` and `3D` folders, then everything else.

## Step 6. Extend the .gitignore

The Python template GitHub gave you does not know about this project.

1. On the repository page, click the file `.gitignore`.
2. Click the pencil icon at the top right to edit it.
3. Scroll to the bottom and add these lines:

```
# Simulation output and datasets
work/
*.h5
*.vti
dataset_fields/

# Local Python environment
.venv/
3D/.venv/

# Files staged for deletion
_to_delete/

# Domain files, shipped as Input_domains.zip instead
2D/Domains/*.dat
```

4. Scroll down, type `Add project-specific ignore rules` as the commit
   message, and click **Commit changes**.

## Step 7. Write the front page

The README is the first thing anyone sees.

1. Click `README.md`, then the pencil icon.
2. Replace the contents with something like the text at the end of this
   document.
3. Commit it.

## Step 8. Check the result

Open your repository in a private browser window, so you see it the way a
stranger does. Confirm that:

* the README renders with headings, not raw text
* `3D/tools/prtlb_2d.py` and `prtlb_3d.py` are both there
* `docs/METHODS_PRT-LB.docx` downloads and opens
* the licence shows in the right-hand sidebar
* nothing named `.venv`, `__pycache__` or `_to_delete` appears anywhere

## Step 9. Changing a file later

Two ways, both in the browser.

**To edit one file:** open it, click the pencil, make the change, click
Commit changes.

**To replace several files:** Add file, then Upload files, then drag the new
versions in. Files with the same name and path are replaced, and GitHub keeps
the old version in the history.

If you find yourself doing this every day, that is the point at which GitHub
Desktop pays for itself. Until then the website is fine.

---

## A suggested README

```markdown
# PRT-DeepONet

A geometry-aware neural operator for pore-scale reactive transport, together
with the lattice Boltzmann simulator that generates its training data.

## What is here

| Folder | Contents |
|---|---|
| `3D/tools/prtlb_2d.py` | The 2D simulator: flow, transport and reactions in one file |
| `3D/tools/prtlb_3d.py` | The 3D simulator, same structure |
| `3D/tools/` | Dataset builders, geometry generation, CompLaB interface, tests |
| `3D/model/` | The DeepONet: training, prediction, evaluation, figures |
| `gui/` | A desktop application that drives everything above |
| `bridge/` | Transfer-learning experiments from 2D to 3D |
| `2D/` | The published 2D predecessor of this work |
| `docs/` | Guides for each part of the code, and the methods paper |

## The simulator

PRT-LB solves Stokes flow through a voxelised pore structure with a
two-relaxation-time lattice Boltzmann scheme, then transports up to four
chemical species through the resulting velocity field with a flux-form finite
volume method, applying biotic (Monod) and abiotic reactions at every step.

`docs/METHODS_PRT-LB.docx` describes the method in full, with equations and
references.

## Running it

    python gui/prt_gui.py

Requires Python 3.10 or later with numpy, scipy, h5py, matplotlib and
torch. The GUI has an install button for the dependencies.

To check that everything works:

    python check_everything.py

## The 2D folder

`2D/` contains the original two-dimensional implementation by Heewon Jung and
is included unmodified for reference. See `2D/LICENSE`.

## Citation

To be added on publication.
```
