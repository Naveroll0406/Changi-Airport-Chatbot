import time
import re
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

class WebsiteScraper:
    def __init__(self, base_urls, max_links=10):
        self.base_urls = base_urls
        self.max_links = max_links
        self.visited = set()
        self.data = []

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(service=Service(), options=chrome_options)

    def extract_text(self, html):
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        texts = [tag.get_text(strip=True) for tag in soup.find_all(['p', 'div', 'span', 'li'])]
        return "\n".join([t for t in texts if len(t) > 30])

    def get_internal_links(self, soup, base_url):
        links = set()
        domain = urlparse(base_url).netloc
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            if domain in urlparse(full_url).netloc and full_url not in self.visited:
                links.add(full_url)
        return list(links)

    def scrape_page(self, url):
        try:
            self.driver.get(url)
            time.sleep(2)
            html = self.driver.page_source
            soup = BeautifulSoup(html, "html.parser")

            text = self.extract_text(html)
            if text:
                self.data.append({"url": url, "content": text})
                print(f"[+] Scraped: {url}")

            # Follow internal links
            for link in self.get_internal_links(soup, url):
                if len(self.visited) >= self.max_links:
                    break
                if link not in self.visited:
                    self.visited.add(link)
                    self.scrape_page(link)

        except Exception as e:
            print(f"[!] Error scraping {url}: {e}")

    def scrape(self):
        for url in self.base_urls:
            if url not in self.visited:
                self.visited.add(url)
                self.scrape_page(url)

        self.driver.quit()
        return self.data

    def save_to_file(self, filename="scraped_data.txt"):
        with open(filename, "w", encoding="utf-8") as f:
            for entry in self.data:
                f.write(f"URL: {entry['url']}\n")
                f.write(entry['content'] + "\n\n" + "-"*80 + "\n")
        print(f"[✓] Saved scraped data to {filename}")


if __name__ == "__main__":
    base_urls = [
        "https://www.changiairport.com/en.html",
        "https://www.jewelchangiairport.com/"
    ]

    scraper = WebsiteScraper(base_urls=base_urls, max_links=15)
    data = scraper.scrape()
    scraper.save_to_file("scraped_content.txt")
