# 🍳 Recipe Log

Recipe Log is a simple, private journal for your cooking:
- Save recipes (ingredients + steps)
- Log each time you cook it (date + notes)
- Optionally update the recipe over time (with change tracking)
- Add “variations” (alternate versions)
- Attach photos to a cook
- Review everything in a chronological “Notebook” and a cross-recipe “Calendar”

This app is built with **Streamlit** and runs on your computer in a web browser.

---

## What you get (features)

### ✅ Recipe Library
- Add a recipe (name, source, ingredients, method)
- Select a recipe from your library
- See the **Original recipe** you first saved

### ✅ Cooking notebook (timeline)
For each recipe, the **Notebook** shows a chronological history of:
- 🍳 **Cook logs** (date cooked + notes)
- ✏️ **Edits** (changes to ingredients/method are tracked with diffs)
- 📝 **Notes** (thoughts without cooking or editing)
- 🧪 **Variation updates** (add/edit/delete variations)

If you attach an edit to a cook log, the Notebook shows that cook log plus *only the diff*.

### ✅ Current version + tracked changes
- View the current recipe
- Toggle “Show changes vs original” to see **tracked changes** (inline highlights)

### ✅ Variations
- Save alternative versions (e.g., “Weeknight version”, “Spicy version”)
- Choose which variation you used when logging a cook

### ✅ Photos
- Attach photos to a specific cook log
- Browse photos by cook and delete any you don’t want

### ✅ Calendar
A day-by-day view of **all cook logs across all recipes**, optionally showing:
- Notes/comments
- Photos
- Newest-first toggle

---

## Where your data is stored (important!)
This app stores everything locally in your project folder:

- `data/recipes_db.json` — all recipes, notes, timeline entries
- `data/photos/<recipe_id>/...` — uploaded photos

**To back up everything**, copy the entire `data/` folder somewhere safe (Dropbox, iCloud Drive, external drive, etc.).

---

# Beginner setup (no coding experience needed)

You’ll do three things:
1. Install Python
2. Download this project
3. Run a few copy/paste commands to start the app

Choose your operating system below.

---

## Step 1 — Install Python

### Windows 10/11
1. Go to the official Python website: https://www.python.org/downloads/
2. Click **Download Python** (the big button)
3. Run the installer
4. **Important:** check ✅ **“Add Python to PATH”** before clicking Install
5. Click Install

To confirm it worked:
- Open **Command Prompt** (Start menu → type “cmd”)
- Type:
  python --version
  You should see something like `Python 3.x.x`.

### macOS
1. Go to: https://www.python.org/downloads/
2. Download the macOS installer (Python 3)
3. Run the installer and follow the prompts

To confirm it worked:
- Open **Terminal** (Command + Space → type “Terminal” → Enter)
- Type:
  python3 --version
  You should see `Python 3.x.x`.

---

## Step 2 — Download this project from GitHub

### Option A (easiest): Download ZIP
1. On the GitHub repo page, click **Code** → **Download ZIP**
2. Unzip it (double-click)
3. You now have a folder like `recipe-log-main` (name may vary)

### Option B (advanced): Clone with git
If you already know git, you can clone the repo. (Not required.)

---

## Step 3 — Open a terminal in the project folder

### Windows
1. Open the project folder in File Explorer
2. Click the address bar (where the folder path is)
3. Type `cmd` and press Enter  
   A Command Prompt opens in that folder.

### macOS
1. Open **Terminal**
2. Type `cd ` (with a trailing space)
3. Drag the project folder into the Terminal window (it will paste the path)
4. Press Enter

---

## Step 4 — Create a “virtual environment” (recommended)
A virtual environment keeps this app’s Python packages separate from everything else on your computer.

### Windows (Command Prompt)
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

### macOS (Terminal)
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

You’ll know the environment is active because you’ll see something like `(.venv)` at the start of your command line.

---

## Step 5 — Run the app
From inside the project folder:

### Windows
streamlit run app.py


### macOS
streamlit run app.py

Streamlit will print a “Local URL” like:
- `http://localhost:8501`

Your browser should open automatically. If it doesn’t, copy/paste that URL into your browser.

---

# How to use the app (workflow)

## 1) Add your first recipe
Sidebar → **Add recipe**
- Enter name (required)
- Optionally add a source (URL or cookbook)
- Paste ingredients and steps
- Save

## 2) Log a cook
Library → select recipe → **New entry** tab → **Log a cook**
- Choose the cooked date
- Optionally select a variation you used
- Write cook notes
- Save

## 3) Improve a recipe over time (tracked)
Library → select recipe → **New entry** tab → **Edit Recipe**
- Update ingredients / method
- Optionally attach the edit to your most recent cook log
- Save

Then check:
- **Notebook** tab to see the history + diffs
- **Current version** tab to see tracked changes vs original

## 4) Save variations
Library → select recipe → **New entry** tab → **Add/Edit variations**
- Add “Weeknight version”, “Gluten-free version”, etc.
- Edit/delete variations any time
- Select a variation when logging a cook

## 5) Add photos
Library → select recipe → **Photos** tab
- Pick a cook entry
- Upload photos (JPG/PNG/WEBP)
- Browse and delete photos

## 6) Browse all cooks across all recipes
Sidebar → **Calendar**
- See cooks grouped by day
- Toggle photos and comments

---

# Troubleshooting

## “python is not recognized” (Windows)
You likely didn’t check **Add Python to PATH** during install.
- Re-run the Python installer and enable it, or install again.

## “streamlit: command not found”
Make sure your virtual environment is active:
- Windows: `.venv\Scripts\activate`
- macOS: `source .venv/bin/activate`

Then reinstall:
```bash
pip install -r requirements.txt
```

## Port already in use
If `localhost:8501` is busy, Streamlit will usually pick another port automatically.
You can also specify one:
```bash
streamlit run app.py --server.port 8502
```

## I want to move the app to another computer
Copy the whole project folder, especially **the `data/` folder**.

---

# Updating the app later
If you download a newer version of the code, keep your existing `data/` folder.

A simple approach:
1. Download the new ZIP
2. Copy your old `data/` folder into the new folder (replace if asked)
3. Run the app again

---

# Uninstall / reset
- To “uninstall”, delete the project folder.
- To reset all your recipes, delete the `data/` folder (this permanently removes your data).

---

## Tech notes (for curious readers)
- Built with: Streamlit
- Data: JSON file in `data/recipes_db.json`
- Photos: stored under `data/photos/<recipe_id>/`
