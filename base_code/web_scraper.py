import requests
from bs4 import BeautifulSoup
import os
import zipfile
import re

def check_valid_version(match: str) -> bool:
    version = match.group(1)
    if version.startswith("3."):
        major, minor = map(int, version.split("."))
        return major >= 3 and minor >= 3
    return False

def get_sitemap(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch sitemap from {url}, status code: {response.status_code}")
        return None

sitemap_url = "https://docs.python.org/sitemap.xml"

def get_python_versions(sitemap_url):
    sitemap_response = get_sitemap(sitemap_url)
    soup = BeautifulSoup(sitemap_response, "xml")
    versions = set()
    for loc in soup.find_all("loc"):
        match = re.search(r"https://docs\.python\.org/([\d.]+)/", loc.text)
        if match and check_valid_version(match):
            versions.add(match.group(1))

    versions = sorted(versions, key=lambda v: list(map(int, v.split('.'))))
    print(f"Found Python versions: {versions}")
    return versions

versions = get_python_versions(sitemap_url)
def download_docs_as_zip(versions):
    base_url = "https://docs.python.org/{}/download.html"
    download_dir = "python_docs"
    os.makedirs(download_dir, exist_ok=True)

    for version in versions:
        url = base_url.format(version)
        response = requests.get(url)

        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        zip_link = None

        for a_tag in soup.find_all("a", href=True):
            if "zip" in a_tag["href"].lower():
                zip_link = a_tag["href"]
                break

        if zip_link:
            zip_url = f"https://docs.python.org/{version}/{zip_link}"
            zip_filename = f"python-{version}-docs.zip"
            zip_path = os.path.join(download_dir, zip_filename)

            print(f"Downloading {zip_url}...")
            zip_response = requests.get(zip_url)
            with open(zip_path, "wb") as f:
                f.write(zip_response.content)
            print(f"Saved {zip_filename} in {download_dir}")

            extract_path = os.path.join(download_dir, f"python-{version}-docs")
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted documentation to {extract_path}")

            os.remove(zip_path)
            print(f"Deleted {zip_filename}")

        else:
            print(f"No ZIP link found for Python {version}")

download_docs_as_zip(versions)