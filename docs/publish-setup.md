# Publish setup — PyPI + npm secrets

Step-by-step guide to set up the secrets that `.github/workflows/publish.yml`
needs, then trigger the publish of `noos` (Python), `noos-langchain`
(Python), and `@triangle-technology/noos` (Node) to their registries.

## Current published state (check first before picking a publish path)

Verify what's already live:

```bash
curl -s -o /dev/null -w "PyPI noos: %{http_code}\n"              https://pypi.org/pypi/noos/json
curl -s -o /dev/null -w "PyPI noos-langchain: %{http_code}\n"    https://pypi.org/pypi/noos-langchain/json
curl -s -o /dev/null -w "npm @triangle-technology/noos: %{http_code}\n" https://registry.npmjs.org/@triangle-technology/noos
```

A `200` means "already published at some version"; `404` means "not yet".
Re-uploading the same version to PyPI or npm will fail by design — bump the
version in the relevant manifest (`bindings/python/pyproject.toml`,
`bindings/python-langchain/pyproject.toml`, or `bindings/node/package.json`)
before pushing a new tag.

## Pick a publish path

- **Path A — first-ever publish of all three** (clean slate on registries):
  set both secrets below, push `v0.x.0` tag, all 6 publish jobs run in
  parallel.
- **Path B — publish only `noos-langchain`** (the two others already live):
  set `PYPI_API_TOKEN` (+ `NPM_TOKEN` if you expect a re-publish of Node
  later), then use **manual workflow_dispatch** with `publish-python=false`,
  `publish-node=false`, `publish-langchain=true`.
- **Path C — re-publish everything with a version bump**: edit all three
  manifests to the next version, push a new `v*` tag, all jobs run.

The workflow has three independent on/off toggles in its manual dispatch
inputs (`publish-python`, `publish-langchain`, `publish-node`) so you can
pick any combination.

## 1. Create a PyPI API token

1. Sign in (or sign up) at https://pypi.org.
2. Go to https://pypi.org/manage/account/token/.
3. Click **Add API token**.
4. **Token name**: `noos-publish-github-actions` (any name works).
5. **Scope**: choose **Entire account (all projects)**.
   - First-time publish of a new project requires account-wide scope;
     PyPI lets you create project-scoped tokens only *after* the first
     release is live.
   - After the first publish, you can rotate to project-scoped tokens
     (`noos` + `noos-langchain` each) and delete this one.
6. Click **Add token**. **Copy the entire token value** — it starts with
   `pypi-` and is shown exactly once.

## 2. Create an npm access token

1. Sign in (or sign up) at https://www.npmjs.com.
2. Enable 2FA if you haven't already (Account Settings → Two-Factor
   Authentication → Authorization and publishing).
3. Go to https://www.npmjs.com/settings/<your-username>/tokens.
4. Click **Generate New Token** → **Classic Token**.
5. **Token name**: `noos-publish-github-actions`.
6. **Type**: choose **Automation**.
   - "Automation" bypasses 2FA prompts — required for CI publish.
   - "Publish" type also works but may prompt for 2FA on each publish.
7. Click **Generate**. **Copy the token value** — it starts with `npm_`.

The first publish of `@triangle-technology/noos` uses this token in
**`--access public`** mode (scoped packages default to private; the
workflow explicitly sets `--access public`).

## 3. Add both secrets to the GitHub repo

1. Browse to https://github.com/Triangle-Technology/noos/settings/secrets/actions.
2. Click **New repository secret** twice:
   - **Name**: `PYPI_API_TOKEN`, **Value**: the `pypi-...` token from step 1.
   - **Name**: `NPM_TOKEN`, **Value**: the `npm_...` token from step 2.
3. Verify both appear under **Repository secrets** with their names.

## 4. Confirm the version

Open `bindings/python/pyproject.toml`, `bindings/python-langchain/pyproject.toml`,
and `bindings/node/package.json`. Each declares its own independent version
string (they are not locked to the Rust crate version).

- `bindings/python` → `version = "0.1.0"`
- `bindings/python-langchain` → `version = "0.1.0"`
- `bindings/node` → `"version": "0.1.0-pre"` (the `-pre` suffix blocks
  this from a stable publish; bump to `"0.1.0"` before tag-push if you
  want Node to publish)

The main Rust crate `Cargo.toml` can be at any version — the publish
workflow does NOT run `cargo publish`. That was already done for
`noos 0.4.0` on crates.io during Session 36.

## 5. Trigger the publish

Two ways:

**(a) Tag push — use only when every package's version has been bumped
to a value not yet on its registry:**

```bash
git tag v0.4.0              # tag name is informational; workflow triggers on any v* tag
git push origin v0.4.0
```

All three ecosystems run in parallel. Jobs where the manifest version
already exists on the registry will fail loudly (PyPI / npm reject
same-version re-upload by design). Bump first, tag second.

**(b) Manual dispatch — lets you skip an ecosystem, e.g. for the
langchain-only path when noos + node are already live at 0.1.0:**

1. Browse to https://github.com/Triangle-Technology/noos/actions/workflows/publish.yml.
2. Click **Run workflow** → branch `main`.
3. Three independent checkboxes: `publish-python`, `publish-langchain`,
   `publish-node`. Each defaults to true.
4. **For the langchain-only first publish right now**: uncheck
   `publish-python` + uncheck `publish-node`, keep `publish-langchain`
   checked, click **Run workflow**.
5. The workflow runs only `langchain-build` + `langchain-publish`. ~1
   minute wallclock.

## 6. Monitor progress

Watch the run at https://github.com/Triangle-Technology/noos/actions/.
The workflow has six jobs that should all go green:

- `python-build` — matrix of 4 platforms (Linux x86_64, macOS x86_64,
  macOS arm64, Windows x86_64). Each produces one abi3-py39 wheel.
- `python-publish` — `twine upload` of all 4 wheels to PyPI `noos`.
- `langchain-build` — one universal Python wheel for `noos-langchain`.
- `langchain-publish` — `twine upload` to PyPI `noos-langchain`
  (depends on `python-publish`).
- `node-build` — matrix of 5 target triples. One prebuilt `.node` binary
  each.
- `node-publish` — `npm publish --access public` of the main package +
  5 platform subpackages to `@triangle-technology/noos`.

Typical wallclock: ~10 minutes total.

## 7. Verify after publish

```bash
# PyPI
pip install noos                  # Rust-backed regulator
pip install noos-langchain        # LangChain adapter
python -c "import noos; print(noos.__version__)"

# npm
npm install @triangle-technology/noos
node -e "console.log(require('@triangle-technology/noos'))"
```

Also check the web pages:

- https://pypi.org/project/noos/
- https://pypi.org/project/noos-langchain/
- https://www.npmjs.com/package/@triangle-technology/noos

## 8. After the first successful publish (security hardening)

1. Rotate the broad-scoped PyPI token to project-scoped tokens:
   - New token with scope **Project: noos** → update `PYPI_API_TOKEN`.
   - Better: use PyPI Trusted Publishers (OIDC) instead of tokens going
     forward. Browse to https://pypi.org/manage/project/noos/settings/publishing/
     to configure GitHub OIDC, then remove `PYPI_API_TOKEN` from repo
     secrets entirely.
2. Delete the initial account-wide token from
   https://pypi.org/manage/account/token/.

## 9. Troubleshooting

**`403 Forbidden` on PyPI twine upload**
- Token expired, project already exists under another owner, or token
  scope too narrow. Regenerate with account-wide scope (step 1).

**`npm ERR! 403 Forbidden — scope not found`**
- Your npm user hasn't been added to the `@triangle-technology` scope.
  Either create the scope first
  (https://www.npmjs.com/settings/<your-user>/packages → Create a new
  organization → `triangle-technology`) or publish under an
  unscoped / personal-scope name by editing `bindings/node/package.json`
  `"name"` field.

**`npm ERR! 402 Payment Required`**
- Scoped packages default to private; the workflow explicitly sets
  `--access public`, so this shouldn't happen. If it does, ensure the
  organization's "package visibility" default allows public publishes.

**CI `maturin develop` fails with "Python not found"**
- The workflow uses `actions/setup-python@v5` which provides Python
  3.12. If this breaks, pin to `python-version: "3.11"` in the relevant
  job and re-push.

**`langchain-publish` fails with "noos version not found on PyPI"**
- The `python-publish` → `langchain-publish` ordering occasionally
  races PyPI's index propagation. Re-run just the `langchain-publish`
  job from the Actions UI after waiting 30 seconds; PyPI's index should
  have caught up by then.

**Already published — need to re-publish a fix**
- PyPI and npm both reject re-uploads of the same version. Bump the
  version in the appropriate `pyproject.toml` / `package.json`, commit,
  push a new tag (`v0.4.1`), and re-run.
