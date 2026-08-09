"""土地覆盖标签映射表 (0-8 <-> 名称)."""

LABEL_NAMES: dict[int, str] = {
    0: "water 水",
    1: "trees 树",
    2: "grass 草",
    3: "flooded_vegetation 被淹植被",
    4: "crops 农作物",
    5: "shrub_and_scrub 灌木与矮树丛",
    6: "built 建筑物",
    7: "bare 空地",
    8: "snow_and_ice 冰雪",
}

# LABEL_NAMES: dict[int, str] = {
#     0: "water",
#     1: "trees",
#     2: "grass",
#     3: "flooded_vegetation",
#     4: "crops",
#     5: "shrub_and_scrub",
#     6: "built",
#     7: "bare",
#     8: "snow_and_ice",
# }

LABEL_IDS: dict[str, int] = {v: k for k, v in LABEL_NAMES.items()}
