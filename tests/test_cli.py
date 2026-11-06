import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chatbot.__main__ import describe_time, is_quit, time_command


def test_time_command():
    assert time_command("/time 23:00") == ("set", 23.0)
    assert time_command("/TIME 6") == ("set", 6.0)
    assert time_command("/time 17:30") == ("set", 17.5)
    assert time_command("/time now") == ("clear", None) and time_command("/time") == ("show", None)
    assert time_command("/time soon") == ("bad", None)
    assert time_command("what time is it") is None and time_command("/time 1 2") is None
    assert "Night" in describe_time(23.0) and describe_time(6.0).startswith("time = 06:00")


def test_bare_goodbyes_quit():
    for t in ("bye", "Bye!", "/quit", "goodbye.", "  EXIT ", "q"):
        assert is_quit(t), t


def test_sentences_containing_a_goodbye_do_not_quit():
    for t in ("bye bye the sad song", "play bye bye blackbird", "I feel sad", ""):
        assert not is_quit(t), t
