import os
import re
import json
import time
import requests

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "oellm_docs")
LINKS_FILE = os.path.join(DOCS_DIR, "external_links.txt")
OUT_DIR = os.path.join(DOCS_DIR, "downloads")

GH_API = "https://api.github.com"
GH_HEADERS = {"Accept": "application/vnd.github.v3+json"}
HF_API = "https://huggingface.co"

RATE_DELAY = 1.0


def safe_filename(name, max_len=120):
    name = re.sub(r"[^\w\s./-]", "", name).strip().replace(" ", "_").replace("/", "_")
    return name[:max_len]


def gh_get(endpoint):
    r = requests.get(f"{GH_API}{endpoint}", headers=GH_HEADERS)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        print("    GitHub rate limited, waiting 60s ...")
        time.sleep(60)
        r = requests.get(f"{GH_API}{endpoint}", headers=GH_HEADERS)
    return r


def save_text(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def parse_links():
    github_links = []
    hf_links = []

    with open(LINKS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("from:"):
                continue
            # Strip URL fragment — never part of the API endpoint, and `requests`
            # drops it on the wire, causing endpoint paths to silently truncate
            # (e.g. /repos/foo/bar#anchor/readme → /repos/foo/bar).
            line = line.split("#", 1)[0]
            if "github.com" in line or "gist.github.com" in line:
                github_links.append(line)
            elif "huggingface.co" in line or "hf.co" in line:
                hf_links.append(line)

    return github_links, hf_links


# --- GitHub ---

def classify_gh(url):
    # Gist
    m = re.match(r"https://gist\.github\.com/([^/]+)/([a-f0-9]+)", url)
    if m:
        return "gist", m.group(1), m.group(2)

    # Issue/PR comment
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/issues/(\d+)", url)
    if m:
        return "issue", f"{m.group(1)}/{m.group(2)}", m.group(3)

    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)", url)
    if m:
        return "pr", f"{m.group(1)}/{m.group(2)}", m.group(3)

    # Specific file
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*?)(?:#.*)?$", url)
    if m:
        return "file", f"{m.group(1)}/{m.group(2)}", f"{m.group(3)}/{m.group(4)}"

    # Tree (directory)
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)(?:/(.*))?", url)
    if m:
        return "repo", f"{m.group(1)}/{m.group(2)}", None

    # Commit
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/commit/([a-f0-9]+)", url)
    if m:
        return "commit", f"{m.group(1)}/{m.group(2)}", m.group(3)

    # Org projects
    m = re.match(r"https://github\.com/orgs/([^/]+)/projects", url)
    if m:
        return "skip", None, None

    # Org/repo listing pages
    m = re.match(r"https://github\.com/orgs/([^/]+)/repositories", url)
    if m:
        return "skip", None, None

    # Org page
    m = re.match(r"https://github\.com/([^/]+)$", url)
    if m:
        return "skip", None, None

    # Repo root
    m = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\?.*)?$", url)
    if m:
        return "repo", f"{m.group(1)}/{m.group(2)}", None

    # Issues listing
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/issues$", url)
    if m:
        return "skip", None, None

    return "unknown", url, None


def download_gh_repo(repo):
    out = os.path.join(OUT_DIR, "github", "repos", safe_filename(repo))
    path = os.path.join(out, "README.md")
    if os.path.exists(path):
        return True

    r = gh_get(f"/repos/{repo}/readme")
    if r.status_code != 200:
        print(f"    no README for {repo}")
        return False

    data = r.json()
    if "content" not in data:
        print(f"    no README content field for {repo} (got keys: {list(data)[:5]})")
        return False
    import base64
    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
    save_text(path, content)
    return True


def download_gh_issue(repo, number):
    out_dir = os.path.join(OUT_DIR, "github", "issues", safe_filename(repo))
    path = os.path.join(out_dir, f"issue_{number}.md")
    if os.path.exists(path):
        return True

    r = gh_get(f"/repos/{repo}/issues/{number}")
    if r.status_code != 200:
        print(f"    failed to fetch issue {repo}#{number}")
        return False

    issue = r.json()
    lines = [
        f"# {issue.get('title', '')}",
        f"**Author:** {issue['user']['login']} | **State:** {issue.get('state', '')} | **Created:** {issue.get('created_at', '')}",
        "",
        issue.get("body", "") or "(no description)",
        "",
    ]

    if issue.get("comments", 0) > 0:
        time.sleep(RATE_DELAY)
        rc = gh_get(f"/repos/{repo}/issues/{number}/comments?per_page=100")
        if rc.status_code == 200:
            lines.append("---")
            lines.append(f"## Comments ({issue['comments']})\n")
            for c in rc.json():
                lines.append(f"**{c['user']['login']}** — {c['created_at']}\n")
                lines.append(c.get("body", "") or "")
                lines.append("\n---\n")

    save_text(path, "\n".join(lines))
    return True


def download_gh_pr(repo, number):
    out_dir = os.path.join(OUT_DIR, "github", "prs", safe_filename(repo))
    path = os.path.join(out_dir, f"pr_{number}.md")
    if os.path.exists(path):
        return True

    r = gh_get(f"/repos/{repo}/pulls/{number}")
    if r.status_code != 200:
        print(f"    failed to fetch PR {repo}#{number}")
        return False

    pr = r.json()
    lines = [
        f"# {pr.get('title', '')}",
        f"**Author:** {pr['user']['login']} | **State:** {pr.get('state', '')} | **Created:** {pr.get('created_at', '')}",
        f"**Base:** {pr['base']['ref']} ← **Head:** {pr['head']['ref']}",
        "",
        pr.get("body", "") or "(no description)",
        "",
    ]

    time.sleep(RATE_DELAY)
    rc = gh_get(f"/repos/{repo}/pulls/{number}/comments?per_page=100")
    if rc.status_code == 200 and rc.json():
        lines.append("---")
        lines.append("## Review Comments\n")
        for c in rc.json():
            lines.append(f"**{c['user']['login']}** on `{c.get('path', '')}` — {c['created_at']}\n")
            lines.append(c.get("body", "") or "")
            lines.append("\n---\n")

    time.sleep(RATE_DELAY)
    rc2 = gh_get(f"/repos/{repo}/issues/{number}/comments?per_page=100")
    if rc2.status_code == 200 and rc2.json():
        lines.append("---")
        lines.append("## Comments\n")
        for c in rc2.json():
            lines.append(f"**{c['user']['login']}** — {c['created_at']}\n")
            lines.append(c.get("body", "") or "")
            lines.append("\n---\n")

    save_text(path, "\n".join(lines))
    return True


def download_gh_file(repo, ref_and_path):
    parts = ref_and_path.split("/", 1)
    if len(parts) < 2:
        return False
    ref, filepath = parts
    out_dir = os.path.join(OUT_DIR, "github", "files", safe_filename(repo))
    out_path = os.path.join(out_dir, safe_filename(filepath))
    if os.path.exists(out_path):
        return True

    raw_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{filepath}"
    r = requests.get(raw_url)
    if r.status_code != 200:
        print(f"    failed to fetch {repo}/{filepath}")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(r.content)
    return True


def download_gh_commit(repo, sha):
    out_dir = os.path.join(OUT_DIR, "github", "commits", safe_filename(repo))
    path = os.path.join(out_dir, f"{sha[:12]}.md")
    if os.path.exists(path):
        return True

    r = gh_get(f"/repos/{repo}/commits/{sha}")
    if r.status_code != 200:
        print(f"    failed to fetch commit {repo}@{sha[:12]}")
        return False

    data = r.json()
    commit = data["commit"]
    lines = [
        f"# {commit['message'].split(chr(10))[0]}",
        f"**Author:** {commit['author']['name']} | **Date:** {commit['author']['date']}",
        "",
        commit["message"],
        "",
        f"**Files changed:** {len(data.get('files', []))}",
    ]

    for f_info in data.get("files", []):
        lines.append(f"\n### {f_info['filename']} ({f_info.get('status', '')})")
        patch = f_info.get("patch", "")
        if patch:
            lines.append(f"```diff\n{patch}\n```")

    save_text(path, "\n".join(lines))
    return True


def download_gh_gist(owner, gist_id):
    out_dir = os.path.join(OUT_DIR, "github", "gists")
    path = os.path.join(out_dir, f"{gist_id}.md")
    if os.path.exists(path):
        return True

    r = gh_get(f"/gists/{gist_id}")
    if r.status_code != 200:
        print(f"    failed to fetch gist {gist_id}")
        return False

    data = r.json()
    lines = [f"# Gist by {owner}: {data.get('description', '')}", ""]

    for fname, fdata in data.get("files", {}).items():
        lines.append(f"## {fname}\n")
        lang = fdata.get("language", "")
        lines.append(f"```{lang.lower() if lang else ''}\n{fdata.get('content', '')}\n```\n")

    save_text(path, "\n".join(lines))
    return True


# OELLM org repos are fully cloned into ~/oellm_org_repos/ via `gh repo clone`, so
# this fetcher skips OELLM org repo-content URLs (README/file/commit). Issues, PRs,
# and gists still come through here — they live in GitHub's database, not the repo.
OELLM_ORG_PREFIX = "OpenEuroLLM/"
OELLM_ORG_SKIP_KINDS = {"repo", "file", "commit"}


def process_github(links):
    seen = set()
    ok, fail, skip = 0, 0, 0

    for url in links:
        kind, repo, extra = classify_gh(url)

        key = (kind, repo, extra)
        if key in seen:
            continue
        seen.add(key)

        time.sleep(RATE_DELAY)

        if kind == "skip" or kind == "unknown":
            skip += 1
            continue

        if repo and repo.startswith(OELLM_ORG_PREFIX) and kind in OELLM_ORG_SKIP_KINDS:
            skip += 1
            continue

        success = False
        if kind == "repo":
            print(f"  [repo] {repo}")
            success = download_gh_repo(repo)
        elif kind == "issue":
            print(f"  [issue] {repo}#{extra}")
            success = download_gh_issue(repo, extra)
        elif kind == "pr":
            print(f"  [pr] {repo}#{extra}")
            success = download_gh_pr(repo, extra)
        elif kind == "file":
            print(f"  [file] {repo}: {extra}")
            success = download_gh_file(repo, extra)
        elif kind == "commit":
            print(f"  [commit] {repo}@{extra[:12]}")
            success = download_gh_commit(repo, extra)
        elif kind == "gist":
            print(f"  [gist] {extra}")
            success = download_gh_gist(repo, extra)

        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nGitHub: {ok} downloaded, {fail} failed, {skip} skipped")


# --- HuggingFace ---

def classify_hf(url):
    url = url.replace("hf.co/", "huggingface.co/")
    path = re.sub(r"https://huggingface\.co/", "", url)

    if path.startswith("blog/") or path.startswith("spaces/") or path.startswith("docs/"):
        return "skip", None

    m = re.match(r"datasets/([^/]+/[^/]+)", path)
    if m:
        return "dataset", m.group(1)

    m = re.match(r"collections/([^/]+/[^/]+)", path)
    if m:
        return "collection", m.group(1)

    m = re.match(r"([^/]+/[^/]+)", path)
    if m and not path.startswith("spaces/"):
        return "model", m.group(1)

    return "skip", None


def download_hf_card(kind, repo_id):
    out_dir = os.path.join(OUT_DIR, "huggingface", kind + "s")
    fname = safe_filename(repo_id)
    path = os.path.join(out_dir, f"{fname}.md")
    if os.path.exists(path):
        return True

    if kind == "dataset":
        url = f"{HF_API}/datasets/{repo_id}/raw/main/README.md"
    elif kind == "model":
        url = f"{HF_API}/{repo_id}/raw/main/README.md"
    else:
        return False

    r = requests.get(url)
    if r.status_code != 200:
        print(f"    no card for {repo_id}")
        return False

    save_text(path, r.text)
    return True


def download_hf_collection(collection_slug):
    out_dir = os.path.join(OUT_DIR, "huggingface", "collections")
    fname = safe_filename(collection_slug)
    path = os.path.join(out_dir, f"{fname}.md")
    if os.path.exists(path):
        return True

    parts = collection_slug.split("/")
    if len(parts) < 2:
        return False

    org = parts[0]
    slug = parts[1]

    r = requests.get(f"https://huggingface.co/api/collections/{org}/{slug}")
    if r.status_code != 200:
        print(f"    failed to fetch collection {collection_slug}")
        return False

    data = r.json()
    lines = [
        f"# {data.get('title', collection_slug)}",
        "",
        data.get("description", "") or "",
        "",
        "## Items\n",
    ]
    for item in data.get("items", []):
        item_type = item.get("type", "")
        item_id = item.get("id", "")
        lines.append(f"- [{item_type}] {item_id}")

    save_text(path, "\n".join(lines))
    return True


def process_huggingface(links):
    seen = set()
    ok, fail, skip = 0, 0, 0

    for url in links:
        kind, repo_id = classify_hf(url)

        if kind == "skip" or repo_id is None:
            skip += 1
            continue

        if (kind, repo_id) in seen:
            continue
        seen.add((kind, repo_id))

        time.sleep(0.5)

        if kind == "collection":
            print(f"  [collection] {repo_id}")
            success = download_hf_collection(repo_id)
        else:
            print(f"  [{kind}] {repo_id}")
            success = download_hf_card(kind, repo_id)

        if success:
            ok += 1
        else:
            fail += 1

    print(f"\nHuggingFace: {ok} downloaded, {fail} failed, {skip} skipped")


def main():
    github_links, hf_links = parse_links()
    print(f"Found {len(github_links)} GitHub links, {len(hf_links)} HuggingFace links\n")

    print("=== GitHub ===")
    process_github(github_links)

    print("\n=== HuggingFace ===")
    process_huggingface(hf_links)

    print("\nDone!")


if __name__ == "__main__":
    main()
