"""
Unit tests for PaddleOCR Section Detection Engine and Metadata Merger.
"""

import json
from src.ingestion.paddle_section_parser import PaddleSectionDetector


SAMPLE_PADDLE_MARKDOWN = """
# Type 2 diabetes in adults: management

1.4 Starting treatment

1.4.1 Lifestyle modifications
Advise lifestyle changes including dietary advice and physical activity.

1.4.2 Pharmacological treatment
Offer metformin as first-line treatment for adults with type 2 diabetes.

1.4.3 Monitoring after initiation
Measure HbA1c levels every 3 to 6 months.
"""

SAMPLE_PADDLE_PAGES = [
    {
        "markdown": {"text": SAMPLE_PADDLE_MARKDOWN, "images": {}},
        "inputImage": "https://example.com/page1.jpg",
        "outputImages": {"layout_det_res": "https://example.com/layout1.jpg"},
        "prunedResult": {
            "parsing_res_list": [
                {
                    "block_id": 1,
                    "block_label": "paragraph_title",
                    "block_content": "1.4 Starting treatment",
                    "block_bbox": [10, 10, 100, 50],
                    "block_polygon_points": [[10, 10], [100, 10], [100, 50], [10, 50]]
                },
                {
                    "block_id": 2,
                    "block_label": "text",
                    "block_content": "1.4.1 Lifestyle modifications\nAdvise lifestyle changes including dietary advice and physical activity.",
                    "block_bbox": [10, 60, 200, 150],
                    "block_polygon_points": [[10, 60], [200, 60], [200, 150], [10, 150]]
                }
            ]
        }
    }
]


def test_paddle_section_detection():
    detector = PaddleSectionDetector(min_chunk_tokens=0)
    result = detector.parse(SAMPLE_PADDLE_MARKDOWN)

    # 1. Document Info Check
    assert "document_info" in result
    assert result["document_info"]["total_sections_detected"] >= 4

    # 2. Hierarchy Tree Check
    tree = result["hierarchy_tree"]
    assert len(tree) > 0

    # 3. Flat Chunks Check
    flat_chunks = result["flat_chunks"]
    assert len(flat_chunks) >= 4

    # Verify section numbers extracted correctly
    section_numbers = [c["section_number"] for c in flat_chunks]
    assert "1.4" in section_numbers
    assert "1.4.1" in section_numbers
    assert "1.4.2" in section_numbers
    assert "1.4.3" in section_numbers

    # Verify hierarchy path breadcrumb
    lifestyle_chunk = next(c for c in flat_chunks if c["section_number"] == "1.4.1")
    assert lifestyle_chunk["section_title"] == "Lifestyle modifications"
    assert "1.4 Starting treatment" in lifestyle_chunk["hierarchy_path"] or "1.4 Starting treatment" in lifestyle_chunk["parent_section"]


def test_metadata_merging():
    detector = PaddleSectionDetector(min_chunk_tokens=0)
    result = detector.parse_from_pages(SAMPLE_PADDLE_PAGES)

    flat_chunks = result["flat_chunks"]
    assert len(flat_chunks) >= 4

    # Check layout_metadata structure
    chunk = flat_chunks[0]
    assert "layout_metadata" in chunk
    assert "page_image_url" in chunk["layout_metadata"]
    assert chunk["layout_metadata"]["page_image_url"] == "https://example.com/page1.jpg"
    assert chunk["layout_metadata"]["layout_image_url"] == "https://example.com/layout1.jpg"


def test_json_serializability():
    detector = PaddleSectionDetector(min_chunk_tokens=0)
    result = detector.parse(SAMPLE_PADDLE_MARKDOWN)

    json_str = json.dumps(result, indent=2)
    parsed_back = json.loads(json_str)
    assert parsed_back["document_info"]["total_sections_detected"] == result["document_info"]["total_sections_detected"]
