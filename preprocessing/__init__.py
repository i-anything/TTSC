"""Catalog preprocessing for the TechJam shopping agent."""

from preprocessing.catalog import (
    CatalogError,
    NormalizedProduct,
    ScanResult,
    canonical_product_text,
    iter_normalized_products,
    normalize_product,
    scan_catalog,
)

__all__ = [
    "CatalogError",
    "NormalizedProduct",
    "ScanResult",
    "canonical_product_text",
    "iter_normalized_products",
    "normalize_product",
    "scan_catalog",
]
