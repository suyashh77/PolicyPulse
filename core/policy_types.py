POLICY_TYPES = {
    "A": {
        "name": "Distance-gated routing",
        "description": "Free in-store return within X miles, $Y fee for mail-in",
        "variables": {
            "distance_miles": [5, 10, 15, 25],
            "fee_usd": [4.95, 7.95, 9.95, 12.95],
        },
        "announcement_template": (
            "Starting {date}, returns within {distance_miles} miles of a store "
            "are free in-store. Mail-in returns will incur a ${fee_usd} fee."
        ),
    },
    "B": {
        "name": "Flat return fee",
        "description": "All mail-in returns cost $Y. In-store free.",
        "variables": {
            "fee_usd": [4.95, 7.95, 9.95, 12.95],
        },
        "announcement_template": (
            "Starting {date}, all mail-in returns will include a ${fee_usd} "
            "return shipping fee. In-store returns remain free."
        ),
    },
    "C": {
        "name": "Keepit gate",
        "description": "Items under $Z get full refund, no return required.",
        "variables": {
            "threshold_usd": [15, 25, 35, 50],
        },
        "announcement_template": (
            "For items under ${threshold_usd}, we'll issue a full refund — "
            "no return needed. Keep it, donate it, or pass it on."
        ),
    },
}
