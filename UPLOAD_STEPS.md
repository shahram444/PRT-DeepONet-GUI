# Uploading PRT-DeepONet to GitHub through the website

The folder is already prepared. It is at

    Desktop\Shahram Files\Geometry Aware 3D\PRT-DeepONet-github

72 files, 14 MB, largest single file 3.1 MB. Nothing in it is a virtual
environment, a cache, a simulation output or one of the 3000 loose domain
files, so it fits inside the web uploader's limits of 100 files per upload and
25 MB per file, in a single go.

It already contains a `README.md`, a `LICENSE`, a `.gitignore` and a
`requirements.txt`, so there is nothing to write by hand afterwards.

---

## Step 1. Turn on hidden files in Windows

`.gitignore` starts with a dot, so File Explorer hides it and Ctrl+A will not
select it.

1. Open `PRT-DeepONet-github` in File Explorer.
2. Click **View** in the toolbar at the top.
3. Choose **Show**, then tick **Hidden items**.
4. `.gitignore` now appears in the list.

## Step 2. Create the repository

1. Go to **github.com**. You are signed in as `shahram444`.
2. Click the green **New** button at the top of the "Top repositories" list on
   the left of the Dashboard.
3. Repository name: `PRT-DeepONet`
4. Description: `Geometry-aware neural operator for pore-scale reactive transport, with a lattice Boltzmann simulator for 2D and 3D.`
5. Select **Public**.
6. Tick **Add a README file**.
7. Leave **Add .gitignore** on "None" and **Choose a license** on "None". Both
   files are already in your folder and yours are better suited to this
   project.
8. Click the green **Create repository**.

## Step 3. Upload

1. On the repository page, click **Add file** at the top right, then
   **Upload files**.
2. Switch to File Explorer, click once inside `PRT-DeepONet-github`, and press
   **Ctrl+A**. This selects the contents, not the folder itself. Check that
   `.gitignore` is highlighted along with the rest.
3. Drag the selection onto the grey box in the browser marked "Drag files
   here to add them to your repository".
4. Wait until the count stops rising. You should see 72 files, with paths such
   as `3D/tools/prtlb_3d.py`, which means the folder structure survived the
   drag.
5. In the "Commit changes" box at the bottom, type
   `Initial upload of the 2D and 3D code`.
6. Leave "Commit directly to the main branch" selected.
7. Click **Commit changes**.

The upload replaces the README that GitHub created in Step 2 with yours. That
is expected.

If GitHub refuses because there are too many files, do it in two rounds: drag
the `2D` and `3D` folders first and commit, then repeat from Step 3 with
everything else.

## Step 4. Check the result

Open `https://github.com/shahram444/PRT-DeepONet` in a private browser window,
so you see it the way a stranger will, and confirm that:

* the README renders as headings and tables, not as raw text
* the licence shows as **MIT** in the right-hand sidebar
* `3D/tools/prtlb_2d.py` and `3D/tools/prtlb_3d.py` are both present
* `docs/METHODS_PRT-LB.docx` downloads and opens
* `2D/Domains` contains only `Input_domains.zip`
* nothing named `.venv`, `__pycache__`, `_to_delete` or `work` appears anywhere

## Step 5. Changing something later

To edit one file, open it on GitHub and click the pencil icon, make the
change, then Commit changes.

To replace several files, use **Add file** then **Upload files** again and
drag the new versions in. A file with the same name and path is replaced, and
the old version stays in the history.

---

## Two notes

**OneDrive.** This project lives inside OneDrive, which is why you saw the
"Rename 1 item?" warning about `write_vti_and_png.py`. The file name is
correct; OneDrive tripped over a copy in progress. Always dismiss that dialog
rather than accepting the rename, because `check_everything.py` looks the
script up by name. Once the code is on GitHub, GitHub is the real backup and
OneDrive is only a convenience.

**Attribution.** The repository is public and includes Heewon's 2D release in
`2D/`. The README and `LICENSE` both say it is redistributed unmodified under
its own terms, but it is still worth a message to him before you make the
repository visible.
