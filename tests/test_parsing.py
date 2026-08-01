import chess

from chessbench.parsing import classify_move, extract_candidate


def start():
    return chess.Board()


def test_simple_legal_move():
    pr = classify_move(start(), "I think the king's pawn is best.\nMOVE: e4")
    assert pr.parse_class == "legal"
    assert pr.move_san == "e4"
    assert pr.move_uci == "e2e4"


def test_last_move_line_wins():
    pr = classify_move(start(), "Maybe MOVE: d4? No, on reflection:\nMOVE: e4")
    assert pr.parse_class == "legal"
    assert pr.move_san == "e4"


def test_no_move_line_is_invalid():
    pr = classify_move(start(), "I resign, this position is hopeless.")
    assert pr.parse_class == "invalid"
    assert pr.error == "no MOVE line found"


def test_garbage_token_is_invalid():
    pr = classify_move(start(), "MOVE: banana")
    assert pr.parse_class == "invalid"


def test_rank_nine_is_invalid():
    pr = classify_move(start(), "MOVE: Qd9")
    assert pr.parse_class == "invalid"


def test_wellformed_but_illegal():
    pr = classify_move(start(), "MOVE: Ke5")
    assert pr.parse_class == "illegal"


def test_illegal_castling_at_start():
    pr = classify_move(start(), "MOVE: O-O")
    assert pr.parse_class == "illegal"


def test_ambiguous_san():
    # Knights on a1 and c1 can both reach b3.
    board = chess.Board("k7/8/8/8/8/8/8/N1N4K w - - 0 1")
    pr = classify_move(board, "MOVE: Nb3")
    assert pr.parse_class == "ambiguous"


def test_promotion():
    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    pr = classify_move(board, "MOVE: a8=Q")
    assert pr.parse_class == "legal"
    assert pr.move_uci == "a7a8q"


def test_trailing_punctuation_stripped():
    pr = classify_move(start(), "MOVE: e4.")
    assert pr.parse_class == "legal"


def test_markdown_wrapping_stripped():
    pr = classify_move(start(), "MOVE: **Nf3**")
    assert pr.parse_class == "legal"
    assert pr.move_san == "Nf3"


def test_bolded_move_label():
    pr = classify_move(start(), "**MOVE**: e4")
    assert pr.parse_class == "legal"
    assert pr.move_san == "e4"


def test_bold_including_colon():
    pr = classify_move(start(), "**MOVE:** e4")
    assert pr.parse_class == "legal"
    assert pr.move_san == "e4"


def test_remove_does_not_match_move_label():
    pr = classify_move(start(), "REMOVE: e5")
    assert pr.parse_class == "invalid"
    assert pr.error == "no MOVE line found"


def test_move_number_prefix_stripped():
    pr = classify_move(start(), "MOVE: 1.e4")
    assert pr.parse_class == "legal"


def test_zero_castling_normalized():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    pr = classify_move(board, "MOVE: 0-0")
    assert pr.parse_class == "legal"
    assert pr.move_san == "O-O"


def test_null_move_is_invalid():
    pr = classify_move(start(), "MOVE: --")
    assert pr.parse_class == "invalid"


def test_lowercase_move_prefix():
    pr = classify_move(start(), "move: e4")
    assert pr.parse_class == "legal"


def test_extract_candidate_none():
    assert extract_candidate("") == (None, None)


def test_fallback_bare_bolded_move():
    # qwen3-style ending: reasoning prose, then the move alone on the last line.
    pr = classify_move(start(), "Given the position, e4 controls the center.\n\nThus, the answer is:\n\n**e4**")
    assert pr.parse_class == "legal"
    assert pr.move_san == "e4"
    assert pr.extraction == "fallback"


def test_fallback_answer_label():
    pr = classify_move(start(), "Long reasoning here.\n\n**Answer:** a3")
    assert pr.parse_class == "legal"
    assert pr.move_san == "a3"
    assert pr.extraction == "fallback"


def test_fallback_rejects_prose():
    # A SAN-shaped token inside a sentence must NOT be scraped out.
    pr = classify_move(start(), "I think e4 is best but I cannot decide.")
    assert pr.parse_class == "invalid"
    assert pr.error == "no MOVE line found"


def test_fallback_never_overrides_protocol():
    # When a MOVE: line exists, the last-line fallback must not compete.
    pr = classify_move(start(), "MOVE: d4\nActually maybe not.\ne4")
    assert pr.parse_class == "legal"
    assert pr.move_san == "d4"
    assert pr.extraction == "protocol"


def test_boxed_latex_answer():
    # Real qwen3:4b habit: "$$ \boxed{Nf3} $$" as the declared answer.
    pr = classify_move(start(), "### Final Answer\n\n$$\n\\boxed{Nf3}\n$$")
    assert pr.parse_class == "legal"
    assert pr.move_san == "Nf3"
    assert pr.extraction == "fallback"


def test_answer_label_with_markdown_colon_outside_bold():
    pr = classify_move(start(), "Reasoning...\n\n**Answer**: d4")
    assert pr.parse_class == "legal"
    assert pr.move_san == "d4"


def test_long_algebraic_with_hyphen():
    pr = classify_move(start(), "**Answer:** d2-d4")
    assert pr.parse_class == "legal"
    assert pr.move_san == "d4"


def test_prose_quoting_protocol_does_not_shadow_answer():
    # Real failure: the model quoted the format instructions mid-essay, and
    # the stray inline match ('-') masked the boxed answer below.
    text = 'The notation "MOVE: -" style is required by the prompt.\n\n$$\n\\boxed{Nf3}\n$$'
    pr = classify_move(start(), text)
    assert pr.parse_class == "legal"
    assert pr.move_san == "Nf3"


def test_refusal_stays_invalid():
    pr = classify_move(start(), "I would need the board position. Please provide it and I'll analyze it for you!")
    assert pr.parse_class == "invalid"


def test_overspecified_long_algebraic_accepted():
    # python-chess parse_san tolerates fully disambiguated forms like e2e4.
    pr = classify_move(start(), "MOVE: e2e4")
    assert pr.parse_class == "legal"
    assert pr.move_san == "e4"
