# Publishing the prepared folder to GitHub

This deliverable is a local source project. No repository or release has been created on an account.

1. Choose the GitHub owner and repository name (suggested: `otoscopy-layerfusion`).
2. Review the project license and upstream notices. Fill author/ORCID, final manuscript title and DOI in `CITATION.cff.template`; rename it to `CITATION.cff` only after completion. The template deliberately does not invent authors or a DOI.
3. Upload the clean source folder, or use Git after creating your own empty GitHub repository:

```bash
git init -b main
git add .
git commit -m "Release reusable otoscopy LayerFusion research code"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Replace OWNER/REPOSITORY with the actual repository you created. The `.gitignore` excludes data, weights, feature caches, generated examples and outputs. The clean export script independently whitelists source directories and checks for unexpected data binaries, private absolute paths and common credential patterns:

```bash
python scripts/export_release.py --destination ../otoscopy-layerfusion-source.zip
```

It uses exclusive creation to avoid overwriting a previous release. The ZIP includes a SHA256 file manifest. Inspect the release before uploading; do not add local patient images or original clinical spreadsheets. Model-weight distribution has its own upstream/license conditions and is not part of this source package.

4. Let the supplied Linux/Windows CI run on GitHub. Only the local environment has been tested at delivery; the workflow configuration is not a claim that remote CI has already run.
5. Tag a fixed version/commit associated with the manuscript. An archival release DOI may be added if you choose an appropriate archive; none is preassigned here.
6. Use the **real, publicly accessible** repository/release URL in the manuscript's code-availability statement. Templates containing OWNER/REPOSITORY are not live study-code links.
