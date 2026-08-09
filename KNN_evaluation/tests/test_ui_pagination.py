"""Tests for ui_pagination pure functions."""
from KNN_evaluation.ui_pagination import (
    PAGE_SIZE,
    paginate_slice,
    total_pages,
    page_controls,
)


class TestPaginateSlice:
    def test_page_size_constant(self):
        assert PAGE_SIZE == 20

    def test_first_page(self):
        items = list(range(45))
        assert paginate_slice(items, 0) == list(range(20))

    def test_second_page(self):
        items = list(range(45))
        assert paginate_slice(items, 1) == list(range(20, 40))

    def test_last_partial_page(self):
        items = list(range(45))
        assert paginate_slice(items, 2) == list(range(40, 45))

    def test_page_out_of_range_returns_empty(self):
        items = list(range(45))
        assert paginate_slice(items, 3) == []
        assert paginate_slice(items, -1) == []

    def test_empty_items(self):
        assert paginate_slice([], 0) == []

    def test_custom_page_size(self):
        items = list(range(10))
        assert paginate_slice(items, 1, page_size=3) == [3, 4, 5]


class TestTotalPages:
    def test_zero_total(self):
        assert total_pages(0) == 1

    def test_exact_multiple(self):
        assert total_pages(40) == 2

    def test_partial(self):
        assert total_pages(41) == 3

    def test_single(self):
        assert total_pages(1) == 1


class TestPageControls:
    def test_first_page(self):
        assert page_controls(0, 45) == (False, True)

    def test_middle_page(self):
        assert page_controls(1, 45) == (True, True)

    def test_last_page(self):
        assert page_controls(2, 45) == (True, False)

    def test_empty_total(self):
        assert page_controls(0, 0) == (False, False)
