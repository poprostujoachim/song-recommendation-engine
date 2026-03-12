# How to Upload to GitHub

## Step 1: Initialize Git Repository

Open terminal in the project folder and run:

```bash
cd C:\Users\jwicz\CascadeProjects\song-recommendation-engine
git init
```

## Step 2: Add Files to Git

```bash
git add .
git commit -m "Initial commit: Song Recommendation Engine with cosine similarity and clustering"
```

## Step 3: Create GitHub Repository

1. Go to https://github.com
2. Click the **+** icon (top right) → **New repository**
3. Fill in:
   - **Repository name:** `song-recommendation-engine`
   - **Description:** "ML-based music recommendation system using cosine similarity and K-means clustering"
   - **Public** or **Private** (your choice)
   - **DO NOT** initialize with README (we already have one)
4. Click **Create repository**

## Step 4: Connect Local to GitHub

GitHub will show you commands. Use these:

```bash
git remote add origin https://github.com/YOUR_USERNAME/song-recommendation-engine.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 5: Verify Upload

Go to your repository URL:
```
https://github.com/YOUR_USERNAME/song-recommendation-engine
```

You should see all your files!

---

## Alternative: Using GitHub Desktop (Easier)

### Step 1: Install GitHub Desktop
Download from: https://desktop.github.com/

### Step 2: Add Repository
1. Open GitHub Desktop
2. Click **File** → **Add Local Repository**
3. Browse to: `C:\Users\jwicz\CascadeProjects\song-recommendation-engine`
4. Click **Add Repository**

### Step 3: Publish to GitHub
1. Click **Publish repository** button
2. Choose repository name
3. Add description
4. Choose Public/Private
5. Click **Publish repository**

Done! ✅

---

## What Gets Uploaded

Your `.gitignore` file ensures these are **NOT** uploaded:
- ✅ Python cache files (`__pycache__`)
- ✅ Virtual environments
- ✅ Model files (`.pkl`)
- ✅ Generated visualizations
- ✅ Large dataset files

Only source code and documentation are uploaded.

---

## After Upload - Add Dataset Instructions

Since the dataset is not uploaded (too large), add this to your GitHub README:

**Users will need to:**
1. Clone your repo
2. Run `python download_real_data.py` to get the dataset
3. Or the system auto-generates sample data on first run

---

## Updating Your Repository Later

After making changes:

```bash
git add .
git commit -m "Description of changes"
git push
```

Or use GitHub Desktop:
1. Review changes
2. Add commit message
3. Click **Commit to main**
4. Click **Push origin**

---

## Common Issues

### "Permission denied (publickey)"
Use HTTPS instead of SSH:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/song-recommendation-engine.git
```

### "Repository already exists"
The repo was already created. Just push:
```bash
git push -u origin main
```

### Large files error
The `.gitignore` should prevent this, but if it happens:
```bash
git rm --cached data/spotify_songs.csv
git commit -m "Remove large dataset file"
git push
```

---

## Making Your Repo Look Professional

### Add Topics/Tags
On GitHub, click **⚙️ Settings** → Add topics:
- `machine-learning`
- `recommendation-system`
- `python`
- `spotify`
- `data-science`
- `clustering`
- `cosine-similarity`

### Add a License
```bash
# Add MIT License (recommended for open source)
```
On GitHub: **Add file** → **Create new file** → Name it `LICENSE` → Choose template

### Add Screenshots
1. Run `streamlit run app.py`
2. Take screenshots
3. Create `screenshots/` folder
4. Add to README with:
```markdown
![Screenshot](screenshots/app.png)
```

---

## Quick Commands Reference

```bash
# First time setup
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/REPO.git
git push -u origin main

# Regular updates
git add .
git commit -m "Your message"
git push

# Check status
git status

# View history
git log --oneline
```
