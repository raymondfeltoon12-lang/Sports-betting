"""Kelly Criterion sizing for a single spread bet, given the model's
predicted probability and the sportsbook's American odds for that side.
"""


def american_to_decimal(american_odds: float) -> float:
    if american_odds > 0:
        return 1 + american_odds / 100
    return 1 + 100 / abs(american_odds)


def kelly_fraction(prob_win: float, american_odds: float, kelly_multiplier: float = 1.0, max_fraction: float = 0.2) -> float:
    """Returns the fraction of bankroll to stake (0 if no positive edge).

    f* = p - (1-p)/b, where b = net decimal odds (profit per unit staked).
    kelly_multiplier < 1.0 gives fractional Kelly (e.g. 0.5 = half-Kelly),
    which is standard practice for reducing variance from probability
    estimation error. max_fraction caps any single bet regardless of the
    formula's output, as a guard against a single overconfident prediction
    wiping out the bankroll.
    """
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    edge = prob_win - (1 - prob_win) / b
    if edge <= 0:
        return 0.0
    return min(edge * kelly_multiplier, max_fraction)
