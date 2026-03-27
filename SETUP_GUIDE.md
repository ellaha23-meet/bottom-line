# AI Tools Extractor — Setup Guide

This guide walks you through every step needed to run the AI Tools Extraction pipeline.

---

## Step 1: Google Cloud Credentials (Gmail + Sheets Access)

You need a `credentials.json` file so the script can read your Gmail and write to Google Sheets.

### 1.1 Create a Google Cloud Project

1. Open your browser and go to **console.cloud.google.com**
2. If this is your first time, accept the Terms of Service
3. At the top of the page, click the **project dropdown** (it may say "Select a project" or show an existing project name)
4. In the popup, click **NEW PROJECT** (top-right of the popup)
5. Enter a project name, e.g. `ai-tools-extractor`
6. Click **CREATE**
7. Wait a few seconds, then make sure the new project is selected in the top dropdown

### 1.2 Enable the Gmail API

1. In the left sidebar, click **APIs & Services** → **Library**
   - If you don't see the sidebar, click the hamburger menu (three horizontal lines) at the top-left
2. In the search bar, type **Gmail API**
3. Click on **Gmail API** in the results
4. Click the blue **ENABLE** button
5. Wait for it to finish enabling

### 1.3 Enable the Google Sheets API

1. Go back to **APIs & Services** → **Library** (use the back button or sidebar)
2. In the search bar, type **Google Sheets API**
3. Click on **Google Sheets API** in the results
4. Click the blue **ENABLE** button
5. Wait for it to finish enabling

### 1.4 Configure the OAuth Consent Screen

Before creating credentials, Google requires you to set up a consent screen:

1. In the left sidebar, click **APIs & Services** → **OAuth consent screen**
2. Select **External** as the user type (unless you have a Google Workspace org), then click **CREATE**
3. Fill in the required fields:
   - **App name**: `AI Tools Extractor` (or anything you like)
   - **User support email**: select your email from the dropdown
   - **Developer contact information**: enter your email address
4. Click **SAVE AND CONTINUE**
5. On the **Scopes** page, click **ADD OR REMOVE SCOPES**
   - Search for and check these two scopes:
     - `https://www.googleapis.com/auth/gmail.readonly`
     - `https://www.googleapis.com/auth/spreadsheets`
   - Click **UPDATE** at the bottom
6. Click **SAVE AND CONTINUE**
7. On the **Test users** page, click **ADD USERS**
   - Enter your own Gmail address (the one with the "AI-NewsLetters" label)
   - Click **ADD**
8. Click **SAVE AND CONTINUE**, then **BACK TO DASHBOARD**

### 1.5 Create OAuth Client Credentials

1. In the left sidebar, click **APIs & Services** → **Credentials**
2. At the top, click **+ CREATE CREDENTIALS** → **OAuth client ID**
3. For **Application type**, select **Desktop app**
4. For **Name**, enter `AI Tools Extractor` (or anything)
5. Click **CREATE**
6. A popup will appear showing your Client ID and Client Secret — click **DOWNLOAD JSON**
7. Rename the downloaded file to exactly **`credentials.json`**
8. Move/copy it into your `bottom-line/` project folder (the same folder as `ai_tools_extractor.py`)

### 1.6 First-Time Authorization

The first time you run the script:
- A browser window will open asking you to sign in to Google
- Select the Google account that has your "AI-NewsLetters" Gmail label
- You may see a warning "This app isn't verified" — click **Advanced** → **Go to AI Tools Extractor (unsafe)**
  - This is normal for personal projects that haven't gone through Google's review
- Grant permission to read Gmail and edit Sheets
- The browser will say "The authentication flow has completed" — you can close it
- A `token.json` file will be saved in your project folder so you won't need to do this again

---

## Step 2: Anthropic API Key (for LLM Analysis)

The script uses Claude to analyze newsletter content and extract AI tools. This requires an Anthropic API key.

### 2.1 Create an Anthropic Account

1. Go to **console.anthropic.com**
2. Click **Sign up** (or **Log in** if you already have an account)
3. Complete the registration process

### 2.2 Add Credits (Pay-as-you-go)

1. Once logged in, click **Settings** (gear icon) in the left sidebar
2. Click **Billing**
3. Click **Add payment method** and enter your card details
4. Add credits (e.g. $5 is plenty to start — a single run costs roughly $0.10–$0.50)

### 2.3 Generate an API Key

1. In the left sidebar, click **API Keys**
2. Click **Create Key**
3. Give it a name, e.g. `ai-tools-extractor`
4. Click **Create Key**
5. **Copy the key immediately** — it starts with `sk-ant-` and you won't see it again

### 2.4 Set the API Key

Before running the script, set it as an environment variable in your terminal:

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Windows (Command Prompt):**
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

To make it permanent (so you don't have to set it every time):
- **Mac/Linux**: Add the `export` line to your `~/.bashrc` or `~/.zshrc` file
- **Windows**: Search for "Environment Variables" in Windows Settings and add it there

---

## Step 3: Google Sheet ID

The script writes results to a Google Sheet. You need to tell it which sheet to use.

### 3.1 Create a New Google Sheet

1. Go to **sheets.google.com**
2. Click the **+** (Blank) button to create a new spreadsheet
3. Give it a name, e.g. `AI Tools Tracker`
4. The script will automatically create the two tabs ("AI Tools Log" and "Field Tools") with headers

### 3.2 Get the Sheet ID

1. Look at the URL in your browser's address bar. It will look like this:
   ```
   https://docs.google.com/spreadsheets/d/1aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789/edit
   ```
2. The Sheet ID is the long string between `/d/` and `/edit`:
   ```
   1aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789
   ```
3. Copy that string

### 3.3 Set the Sheet ID

**Option A — In config.py (recommended):**

Open `config.py` and paste your Sheet ID:
```python
SPREADSHEET_ID = "1aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
```

**Option B — As an environment variable:**
```bash
export SPREADSHEET_ID="1aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789"
```

### 3.4 Ensure Permissions

Make sure the Google account you authorized in Step 1.6 has **edit access** to this spreadsheet. If you created the sheet while logged into the same account, this is automatic.

---

## Step 4: Install Dependencies & Run

### 4.1 Install Python Packages

Open a terminal in the `bottom-line/` project folder and run:

```bash
pip install -r requirements.txt
```

### 4.2 Verify Your Setup

Before running, double-check:

- [ ] `credentials.json` is in the `bottom-line/` folder
- [ ] `ANTHROPIC_API_KEY` environment variable is set
- [ ] `SPREADSHEET_ID` is set (in `config.py` or as env var)
- [ ] Your Gmail account has a label called `AI-NewsLetters`
- [ ] You have edit access to the target Google Sheet

### 4.3 Run the Script

```bash
python ai_tools_extractor.py
```

The script will:
1. Open a browser for Google authorization (first time only)
2. Fetch emails from Gmail labeled "AI-NewsLetters" (last 14 days)
3. Scrape 6 newsletter archive websites (last 14 days)
4. Send content to Claude for analysis
5. Write ranked/categorized AI tools to your Google Sheet

Expected runtime: 3–10 minutes depending on the volume of content.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `credentials.json not found` | Make sure the file is in the same folder as the script |
| `Gmail label not found` | Create a label called exactly `AI-NewsLetters` in Gmail |
| `SPREADSHEET_ID is not set` | Set it in `config.py` or as an environment variable |
| `AuthenticationError` from Anthropic | Check that `ANTHROPIC_API_KEY` is set correctly |
| `insufficient_quota` from Anthropic | Add credits at console.anthropic.com → Billing |
| Browser auth says "app isn't verified" | Click Advanced → Go to app (unsafe) — this is normal |
| `token.json` errors after changing scopes | Delete `token.json` and run again to re-authorize |
