"""Release helper script."""

import subprocess
import sys


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd).decode().strip()


def main():
    action = sys.argv[1]

    if action in ("patch", "minor", "major"):
        run(["uv", "version", "--bump", action])
        version = run(["uv", "version", "--short"])
        run(["git", "add", "pyproject.toml"])
        run(["git", "commit", "-m", f"bump version to {version}"])
        print(f"Bumped to {version} — review with 'git diff HEAD~1' then run 'task tag-push'")

    elif action == "tag-push":
        version = run(["uv", "version", "--short"])
        run(["git", "tag", f"v{version}"])
        run(["git", "push", "--follow-tags"])
        print(f"Tagged and pushed v{version}")

    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
