import urllib.parse
import subprocess
import os

# Hide GTK / Firefox / VA-API logs
os.environ["LIBVA_MESSAGING_LEVEL"] = "0"
os.environ["GTK_A11Y"] = "none"
os.environ["MOZ_DISABLE_GTK_ATK_BRIDGE"] = "1"
os.environ["NO_AT_BRIDGE"] = "1"

browsers = {
    "chrome": "/usr/bin/google-chrome",
    "firefox": "/usr/bin/firefox"
}

# Search URLs
search_endpoints = {
    "youtube": "https://www.youtube.com/results?search_query=",
    "amazon": "https://www.amazon.com/s?k=",
    "wikipedia": "https://en.wikipedia.org/w/index.php?search=",
    "github": "https://github.com/search?q=",
    "reddit": "https://www.reddit.com/search/?q="
}


def open_url(url, browser_path=None):

    try:
        # Open in selected browser silently
        if browser_path:

            subprocess.Popen(
                [browser_path, url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

        # Open in default browser silently
        else:

            subprocess.Popen(
                ["xdg-open", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    except Exception as e:
        print(f"Error: {e}")


while True:
    cmd = input(">>> ").strip()

    if cmd.lower() in ["exit", "quit"]:
        print("Goodbye...We will meet soon\n")
        break

    browser_path = None

    if ":" in cmd:
        browser_name, query = cmd.split(":", 1)
        browser_name = browser_name.strip().lower()
        query = query.strip()

        if browser_name not in browsers:
            print("Browser not supported in current version\n")
            continue

        browser_path = browsers[browser_name]

    else:
        query = cmd.strip()

    if " and search " in query.lower():
        domain, search_term = query.lower().split(" and search ", 1)
        domain = domain.strip()
        search_term = search_term.strip()
        encoded_term = urllib.parse.quote_plus(search_term)

        if domain in search_endpoints:
            target_url = search_endpoints[domain] + encoded_term

        else:
            target_url = (
                f"https://www.google.com/search?q=site:{domain}+{encoded_term}"
            )

        open_url(target_url, browser_path)
        continue

    if "." in query and " " not in query:
        if not query.startswith("http"):
            query = "https://" + query

        open_url(query, browser_path)

    else:
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"https://www.google.com/search?q={encoded_query}"
        open_url(search_url, browser_path)