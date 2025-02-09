from typing import List
from urllib.parse import urlparse, parse_qs, urlencode


class UrlShortener:
    def process_urls(self, urls: List[str]) -> List[str]:
        unique_links = set()

        for url in urls:
            if not url:
                continue

            try:
                cleaned_link = self._clean_and_shorten_url(url)

                if cleaned_link:
                    unique_links.add(cleaned_link)
            except Exception as e:
                print(f"Error processing url {url}: {str(e)}")
                continue

        return sorted(list(unique_links))

    @staticmethod
    def _clean_and_shorten_url(url):
        url = url.strip().lower()

        parsed = urlparse(url)

        params_to_remove = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', '_ga', 'ref', 'source', 'mc_cid', 'mc_eid'
        }

        query_params = parse_qs(parsed.query)
        cleaned_params = {
            k: v[0] if len(v) == 1 else v
            for k, v in query_params.items()
            if k.lower() not in params_to_remove
        }

        clean_url = parsed._replace(
            fragment='',
            query=urlencode(cleaned_params, doseq=True) if cleaned_params else ''
        ).geturl()

        clean_url = clean_url.rstrip('/')

        return clean_url