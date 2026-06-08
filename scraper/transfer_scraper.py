"""
scraper/transfer_scraper.py
Aggregatore notizie calciomercato da fonti multiple.

Fonti:
  - Sky Sport Italia   (HTML scraping)
  - TuttoMercatoWeb    (RSS + HTML)
  - Calciomercato.com  (HTML scraping)
  - Fabrizio Romano    (Twitter/X publico + Substack)
  - Transfermarkt      (news sezione)

Nota su Instagram Stories di Romano:
  Tecnicamente accessibili con `instaloader` + account dummy,
  ma Instagram cambia le API interne frequentemente.
  Tutto quello che Romano posta nelle Stories viene ripubblicato
  su Twitter/X e Telegram entro pochi minuti — quelle fonti
  coprono il 100% dei contenuti rilevanti.
"""
import asyncio
import hashlib
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from config import settings
from db.database import get_db_session
from db.models import NewsItem

logger = logging.getLogger(__name__)

# ─── Pesi delle fonti (usati dal modulo NLP per calcolare probabilità) ───────
SOURCE_WEIGHTS = {
    "romano_twitter":    3.0,   # "here we go" = conferma quasi certa
    "romano_substack":   2.5,
    "sky_sport":         2.0,
    "transfermarkt":     1.8,   # notizie ufficiali (trattative confermate)
    "calciomercato":     1.2,
    "tmw":               1.0,
    "gazzetta":          1.1,
}

# Pattern che indicano una trattativa avanzata / confermata
CONFIRMATION_PATTERNS = [
    r"here we go",
    r"done deal",
    r"accord[oi] trov[ao]t[oi]",
    r"ufficiale",
    r"firma[to]*",
    r"medical[s]?",
    r"visite mediche",
    r"contratto firmato",
    r"annuncio",
]

# Pattern che indicano un semplice rumor
RUMOR_PATTERNS = [
    r"piace",
    r"nel mirino",
    r"si pensa",
    r"potrebbe",
    r"interesse",
    r"sondaggio",
    r"seguito",
    r"idea",
    r"ipotesi",
]


class TransferScraper:
    def __init__(self):
        self._ua      = settings.SCRAPER_USER_AGENT
        self._delay   = settings.SCRAPER_DELAY
        self._session = None

    def _headers(self) -> dict:
        return {
            "User-Agent": self._ua,
            "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def _get(self, url: str, timeout: int = 20) -> Optional[str]:
        """GET sincrono con retry e delay."""
        for attempt in range(3):
            try:
                time.sleep(self._delay)
                with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                    resp = client.get(url, headers=self._headers())
                    resp.raise_for_status()
                    return resp.text
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logger.warning(f"Rate limit su {url}, attendo 30s...")
                    time.sleep(30)
                else:
                    logger.warning(f"HTTP {e.response.status_code} su {url}")
                    break
            except Exception as e:
                logger.warning(f"Tentativo {attempt+1} fallito per {url}: {e}")
                time.sleep(5 * (attempt + 1))
        return None

    # ── Sky Sport ─────────────────────────────────────────────

    def scrape_sky_sport(self) -> list[dict]:
        """
        Scraping notizie mercato da Sky Sport Italia.
        URL: https://sport.sky.it/calcio/calciomercato
        """
        url  = "https://sport.sky.it/calcio/calciomercato"
        html = self._get(url)
        if not html:
            return []

        soup    = BeautifulSoup(html, "lxml")
        articles = []

        # Sky Sport usa card article con classe specifica
        for card in soup.find_all(["article", "div"], class_=re.compile(r"news-item|article-card|story", re.I)):
            title_el = card.find(["h2", "h3", "a"], class_=re.compile(r"title|headline", re.I))
            link_el  = card.find("a", href=True)
            time_el  = card.find(["time", "span"], class_=re.compile(r"time|date|ora", re.I))

            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            url_  = urljoin(url, link_el["href"]) if link_el else ""
            pub   = self._parse_italian_date(time_el.get_text(strip=True) if time_el else "")

            if title and self._is_transfer_news(title):
                articles.append({
                    "source":    "sky_sport",
                    "title":     title,
                    "body":      "",
                    "url":       url_,
                    "published": pub or datetime.utcnow(),
                })

        logger.info(f"Sky Sport: trovati {len(articles)} articoli mercato")
        return articles

    # ── TuttoMercatoWeb (RSS) ─────────────────────────────────

    def scrape_tmw(self) -> list[dict]:
        """
        TuttoMercatoWeb via RSS — più affidabile dello scraping HTML.
        Feed: https://www.tuttomercatoweb.com/rss/
        """
        import xml.etree.ElementTree as ET

        rss_urls = [
            "https://www.tuttomercatoweb.com/rss/news.php",
            "https://www.tuttomercatoweb.com/rss/trattative.php",
        ]
        articles = []

        for rss_url in rss_urls:
            html = self._get(rss_url)
            if not html:
                continue

            try:
                root = ET.fromstring(html)
                channel = root.find("channel")
                if not channel:
                    continue

                for item in channel.findall("item"):
                    title = (item.findtext("title") or "").strip()
                    link  = (item.findtext("link") or "").strip()
                    desc  = (item.findtext("description") or "").strip()
                    pub   = item.findtext("pubDate") or ""

                    if title and self._is_transfer_news(title + " " + desc):
                        articles.append({
                            "source":    "tmw",
                            "title":     title,
                            "body":      BeautifulSoup(desc, "lxml").get_text()[:500],
                            "url":       link,
                            "published": self._parse_rfc_date(pub) or datetime.utcnow(),
                        })

            except ET.ParseError as e:
                logger.warning(f"RSS TMW parse error: {e}")

        logger.info(f"TMW: trovati {len(articles)} articoli mercato")
        return articles

    # ── Calciomercato.com ─────────────────────────────────────

    def scrape_calciomercato(self) -> list[dict]:
        """
        Scraping da Calciomercato.com.
        Usa Playwright perché è JS-heavy.
        """
        try:
            return asyncio.run(self._scrape_calciomercato_async())
        except Exception as e:
            logger.error(f"Errore scraping Calciomercato.com: {e}")
            return []

    async def _scrape_calciomercato_async(self) -> list[dict]:
        from playwright.async_api import async_playwright

        url = "https://www.calciomercato.com/news"
        articles = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=self._ua)
            page    = await context.new_page()

            # Blocca risorse pesanti
            await page.route(
                "**/*.{png,jpg,jpeg,gif,webp,woff,woff2,mp4,mp3}",
                lambda route: route.abort()
            )

            await page.goto(url, wait_until="domcontentloaded", timeout=25000)
            await asyncio.sleep(2)

            html = await page.content()
            await browser.close()

        soup = BeautifulSoup(html, "lxml")

        for card in soup.find_all(["article", "div"], class_=re.compile(r"news|article|item", re.I)):
            title_el = card.find(["h2", "h3", "h4"])
            link_el  = card.find("a", href=True)
            time_el  = card.find("time")

            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            if not title or not self._is_transfer_news(title):
                continue

            articles.append({
                "source":    "calciomercato",
                "title":     title,
                "body":      "",
                "url":       urljoin(url, link_el["href"]) if link_el else "",
                "published": self._parse_html_date(time_el) if time_el else datetime.utcnow(),
            })

        logger.info(f"Calciomercato.com: trovati {len(articles)} articoli")
        return articles

    # ── Fabrizio Romano — Twitter/X pubblico ─────────────────

    def scrape_romano_twitter(self) -> list[dict]:
        """
        Legge i tweet pubblici di Fabrizio Romano via Nitter
        (mirror pubblico di Twitter senza bisogno di API key).

        Istanze Nitter disponibili: nitter.net, nitter.it, nitter.privacydev.net
        Se tutte down → fallback su romano.substack.com
        """
        nitter_instances = [
            "https://nitter.privacydev.net",
            "https://nitter.net",
        ]

        for base in nitter_instances:
            url  = f"{base}/FabrizioRomano"
            html = self._get(url)
            if not html:
                continue

            result = self._parse_nitter_html(html)
            if result:
                logger.info(f"Romano Twitter via {base}: {len(result)} tweet")
                return result

        logger.warning("Romano Twitter: tutti i mirror Nitter non raggiungibili")
        return []

    def _parse_nitter_html(self, html: str) -> list[dict]:
        soup    = BeautifulSoup(html, "lxml")
        tweets  = []

        for tweet_div in soup.find_all("div", class_="timeline-item"):
            content_el = tweet_div.find("div", class_="tweet-content")
            time_el    = tweet_div.find("span", class_="tweet-date")
            link_el    = tweet_div.find("a", class_="tweet-link")

            if not content_el:
                continue

            text = content_el.get_text(strip=True)

            # Filtra solo tweet di mercato
            if not self._is_transfer_news(text):
                continue

            # Detect "here we go"
            hwg = bool(re.search(r"here we go", text, re.I))

            pub_str = ""
            if time_el:
                a = time_el.find("a")
                pub_str = a.get("title", "") if a else time_el.get_text()

            tweets.append({
                "source":    "romano_twitter",
                "title":     text[:200],
                "body":      text,
                "url":       f"https://twitter.com/FabrizioRomano/{link_el['href'].split('/')[-1]}" if link_el else "",
                "published": self._parse_twitter_date(pub_str) or datetime.utcnow(),
                "here_we_go": hwg,
            })

        return tweets

    # ── Fabrizio Romano — Substack ────────────────────────────

    def scrape_romano_substack(self) -> list[dict]:
        """
        Substack di Romano: romano.substack.com/feed (RSS)
        Articoli dettagliati sulle trattative.
        """
        import xml.etree.ElementTree as ET

        url  = "https://romano.substack.com/feed"
        html = self._get(url)
        if not html:
            return []

        articles = []
        try:
            root    = ET.fromstring(html)
            channel = root.find("channel")
            if not channel:
                return []

            for item in channel.findall("item")[:20]:
                title   = (item.findtext("title") or "").strip()
                link    = (item.findtext("link") or "").strip()
                desc    = item.findtext("description") or ""
                pub     = item.findtext("pubDate") or ""

                body = BeautifulSoup(desc, "lxml").get_text()[:1000]

                articles.append({
                    "source":    "romano_substack",
                    "title":     title,
                    "body":      body,
                    "url":       link,
                    "published": self._parse_rfc_date(pub) or datetime.utcnow(),
                    "here_we_go": bool(re.search(r"here we go", title + body, re.I)),
                })

        except ET.ParseError as e:
            logger.warning(f"Substack RSS parse error: {e}")

        logger.info(f"Romano Substack: trovati {len(articles)} articoli")
        return articles

    # ── Transfermarkt news ────────────────────────────────────

    def scrape_transfermarkt(self) -> list[dict]:
        """
        Notizie di trasferimento da Transfermarkt.it
        """
        url  = "https://www.transfermarkt.it/statistiche/neuzugaenge/transfermarkte"
        html = self._get(url)
        if not html:
            return []

        soup     = BeautifulSoup(html, "lxml")
        articles = []

        for row in soup.find_all("tr", class_=re.compile(r"odd|even")):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            player_el = cells[0].find("a")
            from_el   = cells[2].find("a")
            to_el     = cells[3].find("a")
            fee_el    = cells[4] if len(cells) > 4 else None

            if not player_el:
                continue

            player = player_el.get_text(strip=True)
            from_  = from_el.get_text(strip=True) if from_el else ""
            to_    = to_el.get_text(strip=True) if to_el else ""
            fee    = fee_el.get_text(strip=True) if fee_el else ""

            title = f"{player}: da {from_} a {to_} ({fee})"
            articles.append({
                "source":    "transfermarkt",
                "title":     title,
                "body":      title,
                "url":       urljoin(url, player_el.get("href", "")),
                "published": datetime.utcnow(),
                "here_we_go": False,
                "extra": {"player": player, "from": from_, "to": to_, "fee": fee},
            })

        logger.info(f"Transfermarkt: trovati {len(articles)} trasferimenti")
        return articles

    # ── Orchestrator ─────────────────────────────────────────

    def scrape_all_sources(self) -> int:
        """
        Esegue lo scraping di tutte le fonti e salva i nuovi
        articoli nel DB (deduplicazione via hash del titolo).
        Restituisce il numero di nuovi articoli inseriti.
        """
        all_articles = []

        scrapers = [
            ("Romano Twitter",  self.scrape_romano_twitter),
            ("Romano Substack", self.scrape_romano_substack),
            ("Sky Sport",       self.scrape_sky_sport),
            ("TMW",             self.scrape_tmw),
            ("Calciomercato",   self.scrape_calciomercato),
            ("Transfermarkt",   self.scrape_transfermarkt),
        ]

        for name, fn in scrapers:
            try:
                items = fn()
                all_articles.extend(items)
                logger.info(f"  {name}: {len(items)} articoli")
            except Exception as e:
                logger.error(f"  {name} ERRORE: {e}")

        return self._save_articles(all_articles)

    def _save_articles(self, articles: list[dict]) -> int:
        """Salva articoli nel DB, salta i duplicati."""
        count = 0
        with get_db_session() as db:
            for art in articles:
                # Hash per deduplicazione
                h = hashlib.md5(art["title"].encode()).hexdigest()

                # Controlla duplicato
                exists = db.query(NewsItem).filter(
                    NewsItem.title == art["title"][:500],
                ).first()
                if exists:
                    continue

                item = NewsItem(
                    source=art["source"],
                    title=art["title"][:500],
                    body=art.get("body", "")[:5000],
                    url=art.get("url", "")[:500],
                    published=art.get("published", datetime.utcnow()),
                    processed=False,
                )
                db.add(item)
                count += 1

        return count

    # ── Fetch body articolo ───────────────────────────────────

    def fetch_article_body(self, url: str, source: str) -> str:
        """
        Scarica il corpo completo di un articolo per il NLP.
        Utile per estrarre dettagli della trattativa.
        """
        html = self._get(url)
        if not html:
            return ""

        soup = BeautifulSoup(html, "lxml")

        # Rimuovi elementi non-contenuto
        for el in soup.find_all(["script", "style", "nav", "footer", "header"]):
            el.decompose()

        # Cerca il corpo principale
        for selector in ["article", ".article-body", ".content", "main", ".post-content"]:
            el = soup.select_one(selector)
            if el:
                return el.get_text(separator=" ", strip=True)[:3000]

        return soup.get_text(separator=" ", strip=True)[:3000]

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _is_transfer_news(text: str) -> bool:
        """Filtra solo notizie di mercato/trasferimento."""
        keywords = [
            "trasferimento", "mercato", "trattativa", "accordo",
            "rinnovo", "cessione", "acquisto", "prestito",
            "transfer", "deal", "sign", "loan", "contract",
            "here we go", "offerta", "vuole", "cerca",
            "interesse", "piace", "proposta", "clausola",
        ]
        t = text.lower()
        return any(kw in t for kw in keywords)

    @staticmethod
    def _parse_italian_date(text: str) -> Optional[datetime]:
        months = {
            "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
            "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
            "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
        }
        text = text.lower().strip()
        m = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", text)
        if m:
            day, mon, year = m.groups()
            mon_num = months.get(mon)
            if mon_num:
                try:
                    return datetime(int(year), mon_num, int(day))
                except ValueError:
                    pass
        # "ieri" / "oggi"
        if "ieri" in text:
            return datetime.utcnow() - timedelta(days=1)
        if "oggi" in text or "ore" in text:
            return datetime.utcnow()
        return None

    @staticmethod
    def _parse_rfc_date(date_str: str) -> Optional[datetime]:
        """Parsa date RFC 822 (RSS standard)."""
        import email.utils
        try:
            t = email.utils.parsedate_to_datetime(date_str)
            return t.replace(tzinfo=None)
        except Exception:
            return None

    @staticmethod
    def _parse_html_date(time_el) -> Optional[datetime]:
        dt = time_el.get("datetime", "")
        if dt:
            try:
                return datetime.fromisoformat(dt.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_twitter_date(text: str) -> Optional[datetime]:
        """Nitter usa date nel formato "Jan 1, 2024 · 10:30 AM UTC"."""
        try:
            clean = re.sub(r"·.*", "", text).strip()
            return datetime.strptime(clean, "%b %d, %Y")
        except ValueError:
            return None


transfer_scraper = TransferScraper()
