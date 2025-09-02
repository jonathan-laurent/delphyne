"""
Preparing a release consists in upgrading hardcoded version numbers at
several places in the codebase:

- `version` field in `pyproject.toml`.
- `version` field in `vscode-ui/package.json`.
- `git checkout vX.Y.Z` string in the installation instructions from
  `README.md` and `docs/index.md`.
- `mike deploy X.Y latest --push` string in the `Makefile`.

This script also checks whether a proper entry exists for the release in
`CHANGELOG.md` and whether `mkdocs build` runs without error.

Most of this script was generated using Claude Sonnet 4.
"""

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import fire  # type: ignore


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent


def get_current_version(project_root: Path) -> str:
    """Get the current version from pyproject.toml"""
    pyproject_path = project_root / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data["project"]["version"]


def update_version_in_pyproject(version: str, project_root: Path) -> None:
    """Update version in pyproject.toml"""
    pyproject_path = project_root / "pyproject.toml"

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    old_version = data["project"]["version"]

    # Read as text and do string replacement since tomllib is read-only
    with open(pyproject_path, "r") as f:
        content = f.read()

    # Replace the version line
    version_pattern = r'version = "[^"]*"'
    new_content = re.sub(version_pattern, f'version = "{version}"', content)

    with open(pyproject_path, "w") as f:
        f.write(new_content)

    print(f"Updated pyproject.toml: {old_version} -> {version}")


def update_version_in_package_json(version: str, project_root: Path) -> None:
    """Update version in vscode-ui/package.json"""
    package_json_path = project_root / "vscode-ui" / "package.json"

    with open(package_json_path, "r") as f:
        data = json.load(f)

    old_version = data["version"]
    data["version"] = version

    with open(package_json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Updated vscode-ui/package.json: {old_version} -> {version}")


def update_checkout_instructions(version: str, project_root: Path) -> None:
    """Update git checkout instructions in README.md and docs/index.md"""
    files_to_update = [
        project_root / "README.md",
        project_root / "docs" / "index.md",
    ]

    for file_path in files_to_update:
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping")
            continue

        with open(file_path, "r") as f:
            content = f.read()

        # Pattern to match git checkout vX.Y.Z
        pattern = r"git checkout v\d+\.\d+\.\d+"
        replacement = f"git checkout v{version}"

        old_matches = re.findall(pattern, content)
        new_content = re.sub(pattern, replacement, content)

        if old_matches:
            with open(file_path, "w") as f:
                f.write(new_content)
            print(
                f"Updated {file_path.relative_to(project_root)}: {old_matches[0]} -> {replacement}"
            )
        else:
            print(
                f"Warning: No git checkout pattern found in {file_path.relative_to(project_root)}"
            )


def update_makefile_mike_deploy(version: str, project_root: Path) -> None:
    """Update mike deploy command in Makefile with major.minor version"""
    makefile_path = project_root / "Makefile"

    if not makefile_path.exists():
        print("Warning: Makefile not found, skipping")
        return

    # Extract major.minor from version (e.g., "0.7.0" -> "0.7")
    version_parts = version.split(".")
    if len(version_parts) < 2:
        print(f"Warning: Invalid version format {version}, expecting X.Y.Z")
        return

    major_minor = f"{version_parts[0]}.{version_parts[1]}"

    with open(makefile_path, "r") as f:
        content = f.read()

    # Pattern to match mike deploy X.Y latest --push
    pattern = r"mike deploy \d+\.\d+ latest --update-aliases --push"
    replacement = f"mike deploy {major_minor} latest --update-aliases --push"

    old_matches = re.findall(pattern, content)
    new_content = re.sub(pattern, replacement, content)

    if old_matches:
        with open(makefile_path, "w") as f:
            f.write(new_content)
        print(f"Updated Makefile: {old_matches[0]} -> {replacement}")
    else:
        print("Warning: No mike deploy pattern found in Makefile")


def check_changelog_entry(version: str, project_root: Path) -> bool:
    """Check if changelog has an entry for the given version"""
    changelog_path = project_root / "CHANGELOG.md"

    if not changelog_path.exists():
        print("Error: CHANGELOG.md not found")
        return False

    with open(changelog_path, "r") as f:
        content = f.read()

    # Look for version pattern in changelog
    version_pattern = f"## Version {re.escape(version)}"
    if re.search(version_pattern, content):
        print(f"✓ Found changelog entry for version {version}")
        return True
    else:
        print(f"✗ No changelog entry found for version {version}")
        return False


def check_mkdocs_build(project_root: Path) -> bool:
    """Check if mkdocs build runs without error"""
    try:
        subprocess.run(
            ["mkdocs", "build"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        print("✓ mkdocs build successful")
        return True
    except subprocess.CalledProcessError as e:
        print("✗ mkdocs build failed:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("✗ mkdocs command not found. Please ensure mkdocs is installed.")
        return False


class PrepareReleaseCLI:
    """CLI tool for preparing a Delphyne release"""

    def prepare(self, version: str, *, check_only: bool = False) -> None:
        """
        Prepare a release by updating version numbers and checking prerequisites.

        Args:
            version: The version number to release (e.g., "0.7.0")
            check_only: If True, only perform checks without modifying files
        """
        project_root = get_project_root()
        success = True

        print(f"Preparing release v{version}")
        print(f"Project root: {project_root}")
        print()

        if not check_only:
            try:
                update_version_in_pyproject(version, project_root)
                update_version_in_package_json(version, project_root)
                update_checkout_instructions(version, project_root)
                update_makefile_mike_deploy(version, project_root)
                print()
            except Exception as e:
                print(f"Error updating files: {e}")
                success = False

        # Perform checks
        print("Performing checks:")
        if not check_changelog_entry(version, project_root):
            success = False

        if not check_mkdocs_build(project_root):
            success = False

        print()
        if success:
            if check_only:
                print("✓ All checks passed")
            else:
                print("✓ Release preparation completed successfully")
        else:
            print("✗ Some checks failed")
            sys.exit(1)

    def check(self, version: str) -> None:
        """
        Check if the project is ready for release without modifying files.

        Args:
            version: The version number to check (e.g., "0.7.0")
        """
        self.prepare(version, check_only=True)

    def current_version(self) -> None:
        """
        Print the current version number from pyproject.toml to stdout.
        """
        project_root = get_project_root()
        version = get_current_version(project_root)
        print(version)


def main():
    fire.Fire(PrepareReleaseCLI)  # type: ignore


if __name__ == "__main__":
    main()
