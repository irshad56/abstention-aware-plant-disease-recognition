"""Trusted agricultural evidence source registry and domain whitelists."""

from typing import List, Set

TRUSTED_DOMAINS: Set[str] = {
    ".edu",
    ".gov",
    "apsnet.org",
    "cabi.org",
    "fao.org",
    "plantvillage.psu.edu",
    "extension.psu.edu",
    "ipm.ucanr.edu",
    "edis.ifas.ufl.edu",
    "ohioline.osu.edu",
    "extension.purdue.edu",
    "crops.extension.iastate.edu",
    "cropprotectionnetwork.org",
    "usablight.org",
    "njaes.rutgers.edu",
    "vegpath.plantpath.wisc.edu",
    "plantpathology.ca.uky.edu",
    "content.ces.ncsu.edu",
    "extension.uga.edu",
    "extension.umn.edu",
    "missouribotanicalgarden.org",
    # Authoritative Peer-Reviewed Research Literature Repositories (Q1 Journals)
    "doi.org",
    "nature.com",
    "springer.com",
    "sciencedirect.com",
    "wiley.com",
    "frontiersin.org",
    "academic.oup.com",
    "ncbi.nlm.nih.gov",
}


def is_trusted_url(url: str, allowed_domains: List[str] = None) -> bool:
    """Validates whether a URL originates from an authoritative agricultural domain."""
    if not url or not url.startswith(("http://", "https://")):
        return False

    url_lower = url.lower()
    domains_to_check = allowed_domains if allowed_domains else TRUSTED_DOMAINS

    for domain in domains_to_check:
        if domain.startswith("."):
            # Suffix match, e.g. .edu or .gov
            # Extract host
            parts = url_lower.split("/")[2].split("?")[0]
            if parts.endswith(domain) or f"{domain}:" in parts:
                return True
        else:
            if domain in url_lower:
                return True

    return False
