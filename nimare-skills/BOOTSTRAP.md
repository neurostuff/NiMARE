# Bootstrap: moving this into its own repository

This directory is the complete contents of the intended **`neurostuff/nimare-skills`**
repository. It is staged here only because the session that built it could not create a
new GitHub repository (its GitHub App token is scoped to `neurostuff/NiMARE` and has no
repo-creation permission).

**This directory is not meant to be merged into NiMARE.** Extract it, then delete the
branch.

## One-time setup

```bash
# 1. Create the empty repo (GitHub UI, or gh CLI)
gh repo create neurostuff/nimare-skills --public \
  --description "Agent skills for neuroimaging meta-analysis with NiMARE"

# 2. Get this directory out of the staging branch
git clone https://github.com/neurostuff/NiMARE /tmp/nimare-staging
cd /tmp/nimare-staging
git checkout claude/eager-rubin-qxigmu

# 3. Push it as the new repo's initial commit
cd nimare-skills
rm BOOTSTRAP.md
git init -b main
git add -A
git commit -m "Initial commit: NiMARE agent skills"
git remote add origin https://github.com/neurostuff/nimare-skills
git push -u origin main

# 4. Clean up the staging branch
cd /tmp/nimare-staging
git push origin --delete claude/eager-rubin-qxigmu
```

## Verify the install paths work

The README documents:

```
/plugin marketplace add neurostuff/nimare-skills
/plugin install nimare-skills@nimare-skills
```

Both resolve once the repo is public at that path, because
`.claude-plugin/marketplace.json` lists a single plugin whose `source` is `./`.

## Before announcing it

- `python skills/nimare-dataset-curation/scripts/audit_independence.py --help`
- `python skills/nimare-cbma-ale/scripts/run_ale.py --help`
- Both scripts were run end to end against NiMARE 0.22 and the bundled
  `semantic_knowledge_children.txt` / `semantic_relatedness_children.txt` resources.
- Consider adding a CI job that runs both scripts against those bundled files, so the
  skills break loudly when the NiMARE API moves.
