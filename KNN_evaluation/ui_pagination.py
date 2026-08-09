"""WebUI 数据导入区预览分页辅助纯函数.

独立于 NiceGUI，便于单元测试。所有函数为纯函数，不持有 UI 状态。
"""

PAGE_SIZE = 20


def paginate_slice(items: list, page: int, page_size: int = PAGE_SIZE) -> list:
    """返回 items[page*page_size : (page+1)*page_size].

    Args:
        items: 待分页的条目列表。
        page: 从 0 开始的页码。
        page_size: 每页条目数。

    Returns:
        当前页切片；items 为空或 page 越界时返回空列表。
    """
    if not items:
        return []
    start = page * page_size
    if start < 0 or start >= len(items):
        return []
    return items[start:start + page_size]


def total_pages(total: int, page_size: int = PAGE_SIZE) -> int:
    """计算总页数.

    Args:
        total: 条目总数。
        page_size: 每页条目数。

    Returns:
        总页数；total == 0 时返回 1。
    """
    if total <= 0:
        return 1
    return (total + page_size - 1) // page_size


def page_controls(page: int, total: int, page_size: int = PAGE_SIZE) -> tuple[bool, bool]:
    """推导翻页按钮可用态.

    Args:
        page: 从 0 开始的当前页码。
        total: 条目总数。
        page_size: 每页条目数。

    Returns:
        (can_prev, can_next)。
    """
    can_prev = page > 0
    can_next = (page + 1) * page_size < total
    return can_prev, can_next
