from barrikada import PIPipeline


def test_public_api_exposes_pipeline() -> None:
    assert PIPipeline.__name__ == "PIPipeline"
